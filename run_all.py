import subprocess, sys
for step in ["download", "etl", "ingest", "evaluate"]:
    print(f"\n=== {step} ==="); subprocess.run([sys.executable, f"src/{step}.py"], check=True)
print("\nDone. Launch dashboard:  streamlit run app.py")
