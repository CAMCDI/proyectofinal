import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append('/home/aza/Documentos/simulacion/proyectoFinal')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from file_manager.ml_utils import MLManager

def test_tasks():
    print("🚀 Verifying 6 ML Functional Paths (05-10)...")
    
    # 1. Spam (Notebook 05)
    print("\n[1a] Testing Spam (email)...")
    with open('test_spam.txt', 'w') as f: f.write("Subject: FREE MONEY WIN PRIZE!!!")
    res = MLManager.analyze_05_spam('test_spam.txt')
    print(f"Result: {res.get('result')} | Success: {res.get('success')}")
    os.remove('test_spam.txt')

    print("\n[1b] Testing Spam (TREC index)...")
    with open('index', 'w') as f: 
        f.write("spam ../data/inmail.1\nham ../data/inmail.2\nspam ../data/inmail.3")
    res = MLManager.analyze_05_spam('index')
    print(f"Result: {res.get('result')} | Success: {res.get('success')}")
    os.remove('index')

    # 2. Dummy Data for ARFF Tasks
    dummy_arff = 'test_data.arff'
    with open(dummy_arff, 'w') as f:
        f.write("@relation test\n")
        f.write("@attribute col1 numeric\n")
        f.write("@attribute protocol_type {tcp, udp, icmp}\n")
        f.write("@attribute same_srv_rate numeric\n")
        f.write("@attribute dst_host_srv_count numeric\n")
        f.write("@attribute dst_host_same_srv_rate numeric\n")
        f.write("@attribute class {normal, anomaly}\n")
        f.write("@data\n")
        # Generate 10 rows for stratification
        rows = [
            "1,tcp,0.5,10,0.5,normal",
            "2,tcp,0.6,11,0.6,normal",
            "3,udp,0.1,5,0.1,anomaly",
            "4,udp,0.2,6,0.2,anomaly",
            "5,icmp,0.0,2,0.0,anomaly",
            "6,icmp,0.0,3,0.0,anomaly",
            "7,tcp,0.7,12,0.7,normal",
            "8,tcp,0.8,13,0.8,normal",
            "9,udp,0.3,7,0.3,anomaly",
            "10,udp,0.4,8,0.4,anomaly"
        ]
        f.write("\n".join(rows))

    # Task Tests
    tasks = [
        ("Visualización (06)", MLManager.analyze_06_viz),
        ("División (07)", MLManager.analyze_07_split),
        ("Preparación (08)", MLManager.analyze_08_09_prep),
        ("Pipelines (09)", MLManager.analyze_08_09_prep),
        ("Evaluación (10)", MLManager.analyze_10_eval),
    ]

    for name, func in tasks:
        print(f"\nTesting {name}...")
        res = func(dummy_arff)
        if res['success']:
            print(f"Result: {res['result']} | Success: {res['success']} | Graphs: {len(res.get('graphics', []))}")
        else:
            print(f"FAILED: {res.get('error')}")

    os.remove(dummy_arff)

if __name__ == "__main__":
    test_tasks()
    print("\n✅ Verification Complete.")
