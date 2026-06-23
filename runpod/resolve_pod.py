#!/usr/bin/env python3
"""Resolve a RunPod pod by NAME -> wait for RUNNING + SSH -> write a pod json.

Works around a RunPod REST bug where POST /pods creates the pod but returns 500
with no id (so `launch_pod.py launch` can't capture it). Find it by name instead.

  python runpod/resolve_pod.py --name neucodec-44k-d16 --out runpod/pod-d16.json
"""
import argparse
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import launch_pod as L  # reuse api(), extract_ssh(), load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ssh-key", default="~/.ssh/id_rsa")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    L.load_dotenv()

    # newest pod with this exact name
    pods = [p for p in L.api("GET", "/pods") if p.get("name") == a.name]
    if not pods:
        sys.exit(f"no pod named {a.name!r}")
    pod = pods[-1]
    pod_id = pod["id"]
    print(f"[resolve] {a.name} -> pod_id={pod_id}")

    deadline = time.time() + a.timeout
    while time.time() < deadline:
        pod = L.api("GET", f"/pods/{pod_id}")
        status = pod.get("desiredStatus") or pod.get("status")
        ip, port = L.extract_ssh(pod)
        if status == "RUNNING" and ip and port:
            import json
            meta = {"pod_id": pod_id, "name": a.name, "ip": ip, "ssh_port": port,
                    "ssh_key": a.ssh_key, "cost_per_hr": pod.get("costPerHr")}
            with open(a.out, "w") as f:
                json.dump(meta, f, indent=2)
            key = a.ssh_key.replace("~", __import__("os").path.expanduser("~"))
            print(f"[resolve] RUNNING ip={ip} port={port} -> {a.out}")
            print(f"[resolve] ssh -p {port} -i {key} -o StrictHostKeyChecking=no "
                  f"-o UserKnownHostsFile=/dev/null root@{ip}")
            return
        print(f"  … status={status} ip={ip} port={port} ({int(deadline-time.time())}s left)")
        time.sleep(10)
    sys.exit("[resolve] timed out waiting for SSH")


if __name__ == "__main__":
    main()
