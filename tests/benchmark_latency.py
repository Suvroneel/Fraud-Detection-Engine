"""
Latency Benchmark Script
Tests the scoring API and reports p50, p95, p99 latency.
Run this after docker-compose up to validate performance targets.
"""

import time
import json
import random
import statistics
import urllib.request
import urllib.error

API_URL = 'http://localhost:8000/score'

SAMPLE_TRANSACTIONS = [
    {
        'TransactionAmt' : 50.0,
        'ProductCD'      : 'W',
        'card1'          : 12345,
        'card4'          : 'visa',
        'card6'          : 'debit',
        'P_emaildomain'  : 'gmail.com',
        'DeviceType'     : 'desktop',
    },
    {
        'TransactionAmt' : 1500.0,
        'ProductCD'      : 'C',
        'card1'          : 99999,
        'card4'          : 'mastercard',
        'card6'          : 'credit',
        'P_emaildomain'  : 'anonymous.com',
        'DeviceType'     : 'mobile',
    },
    {
        'TransactionAmt' : 250.0,
        'ProductCD'      : 'H',
        'card1'          : 54321,
        'P_emaildomain'  : 'yahoo.com',
        'DeviceType'     : 'desktop',
    },
]


def send_request(payload: dict) -> float:
    """Send a single scoring request and return latency in ms."""
    data = json.dumps(payload).encode('utf-8')
    req  = urllib.request.Request(
        API_URL,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()
    except urllib.error.URLError as e:
        print(f'Request failed: {e}')
        return -1.0
    end = time.perf_counter()
    return (end - start) * 1000


def run_benchmark(n_requests: int = 200):
    print(f'Running latency benchmark with {n_requests} requests...')
    print(f'Target: p50 < 250ms, p95 < 500ms, p99 < 500ms\n')

    latencies = []
    for i in range(n_requests):
        tx = random.choice(SAMPLE_TRANSACTIONS).copy()
        tx['TransactionAmt'] = random.uniform(10, 2000)
        tx['TransactionID']  = i

        lat = send_request(tx)
        if lat > 0:
            latencies.append(lat)

        if (i + 1) % 50 == 0:
            print(f'  Completed {i+1}/{n_requests} requests...')

    if not latencies:
        print('No successful requests recorded.')
        return

    latencies.sort()
    n = len(latencies)

    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]

    print('\nLatency Benchmark Results')
    print(f'Total requests    : {n_requests}')
    print(f'Successful        : {n}')
    print(f'Mean latency      : {statistics.mean(latencies):.1f} ms')
    print(f'Median (p50)      : {p50:.1f} ms')
    print(f'p95               : {p95:.1f} ms')
    print(f'p99               : {p99:.1f} ms')
    print(f'Max               : {max(latencies):.1f} ms')

    print('\nTarget Check:')
    print(f'p50 < 250ms : {"PASS" if p50 < 250 else "FAIL"} ({p50:.1f}ms)')
    print(f'p95 < 500ms : {"PASS" if p95 < 500 else "FAIL"} ({p95:.1f}ms)')
    print(f'p99 < 500ms : {"PASS" if p99 < 500 else "FAIL"} ({p99:.1f}ms)')


if __name__ == '__main__':
    run_benchmark(200)
