import sys
from pathlib import Path
from datetime import datetime
from threading import Thread
from subprocess import Popen, PIPE
from concurrent.futures import ThreadPoolExecutor, as_completed

N = 50

scores = [0] * N

def read_stream(name, in_file, out_file):
    for line in in_file:
        print(f"[{name}] {line.strip()}", file=out_file)
        try:
            scores[name] = int(line.strip().split()[-1])
        except:
            pass

def run(cmd, name):
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    stdout_thread = Thread(target=read_stream, args=(name, proc.stdout, sys.stdout))
    stderr_thread = Thread(target=read_stream, args=(name, proc.stderr, sys.stderr))
    stdout_thread.start()
    stderr_thread.start()
    proc.wait()
    return proc

# out_dir = Path("out") / datetime.now().isoformat()
# out_dir.mkdir()
# with ThreadPoolExecutor(7) as executor:
#     futures = []
#     for i in range(N):
#         out_file = out_dir / f"{i:04d}.txt"
#         cmd = f"./tools/target/release/tester ./a.out < ./tools/in/{i:04d}.txt > {out_file}"
#         futures.append(executor.submit(run, cmd, i))
#     as_completed(futures)
#
# print(f"Mean Score = {sum(scores) / len(scores)}")

for i in range(N):
    cmd = f"time ./a.out < ./tools/in/{i:04d}.txt"
    run(cmd, f"{i}")