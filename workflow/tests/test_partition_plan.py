#!/usr/bin/env python3
"""Check plan_partition() against the hardware it claims to model.

    python3 workflow/tests/test_partition_plan.py     # expect ALL PASS

The numbers in HPC_PARTITIONS came from real allocations (lscpu + numactl
--hardware on cn03/cn15/cn21, recorded in research_backup/setup/
hpc_compute_*.md) and from sinfo / sacctmgr on login.iitgn.ac.in. Nothing
here can re-measure the cluster, so what it does check is that the arithmetic
built on those numbers stays self-consistent: that every layout fills exactly
one node, that no plan exceeds its QoS cap, and -- the one that would
actually bite -- that the literal defaults in the HPC profile still equal
what plan_partition() computes for them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hpc_config                                              # noqa: E402

failures = []


def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")
        failures.append(label)


print("npn * cpt fills exactly one node (no idle cores, no oversubscription)")
for name, p in hpc_config.HPC_PARTITIONS.items():
    if p.cores_per_node > p.cpu_cap:
        continue                    # test: the cap, not the node, is the limit
    plan = hpc_config.plan_partition(name)
    check(f"{name}: {plan['npn']} x {plan['cpt']} == {p.cores_per_node} cores",
          plan["npn"] * plan["cpt"], p.cores_per_node)

print("\none MPI rank per NUMA domain")
for name, p in hpc_config.HPC_PARTITIONS.items():
    check(f"{name}: npn == {p.numa_per_node} NUMA domains",
          hpc_config.plan_partition(name)["npn"], p.numa_per_node)

print("\nno plan exceeds its QoS cap (would sit pending on "
      "QOSMaxCpuPerUserLimit)")
for name, p in hpc_config.HPC_PARTITIONS.items():
    for nodes in (1, 2, 3):
        plan = hpc_config.plan_partition(name, nodes=nodes)
        total = (plan["nodes"] * plan["npn"] * plan["cpt"]
                 * plan["max_concurrent_jobs"])
        check(f"{name} @ {nodes} node(s): {total} <= {p.cpu_cap} cores",
              total <= p.cpu_cap, True)

print("\ntest's 8-core QoS cap shrinks the layout instead of overrunning it")
plan = hpc_config.plan_partition("test")
check("test: 2 ranks x 4 threads", (plan["npn"], plan["cpt"]), (2, 4))
check("test: 1 concurrent job", plan["max_concurrent_jobs"], 1)

print("\nthe HPC profile's literal defaults still equal "
      "plan_partition('medium', nodes=1)")
want = hpc_config.plan_partition("medium", nodes=1)
got = {k: getattr(hpc_config.HPC, k) for k in want}
check("HPC defaults", got, want)

print("\nhosts without a profiled table are not silently given hpc's")
for host in (hpc_config.PARAM, hpc_config.LOCAL):
    check(f"{host.name}: no partition table", host.partitions, {})

print("\nan unknown queue name is refused, not guessed at")
try:
    hpc_config.plan_partition("gpu")
except ValueError:
    print("  PASS  plan_partition('gpu') raises ValueError")
else:
    print("  FAIL  plan_partition('gpu') did not raise")
    failures.append("unknown partition")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
