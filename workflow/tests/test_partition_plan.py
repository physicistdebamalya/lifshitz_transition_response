#!/usr/bin/env python3
"""Check plan_partition() against the hardware it claims to model.

    python3 workflow/tests/test_partition_plan.py     # expect ALL PASS

The numbers in HPC_PARTITIONS came from real allocations (lscpu + numactl
--hardware on cn03/cn15/cn21, recorded in research_backup/setup/
hpc_compute_*.md) and from sinfo / sacctmgr on login.iitgn.ac.in. The numbers
in PARAM_PARTITIONS came the same way -- lscpu on cn003/cn023/cn110/hm002,
recorded in research_backup/setup/supercomputing_cluster/param/ -- and from
sinfo / sacctmgr on login02/login03.paramananta.iitgn.ac.in. Nothing here can
re-measure either cluster, so what it does check is that the arithmetic built
on those numbers stays self-consistent: that every layout fills exactly one
node, that no plan exceeds its QoS cap, and -- the one that would actually
bite -- that the literal defaults in each HostConfig still equal what
plan_partition() computes for them.
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


# Both profiled clusters get the same checks. Looping over this instead of
# writing HPC_PARTITIONS and PARAM_PARTITIONS out twice is what keeps them
# from drifting the way the two npn/cpt comment blocks did before.
CLUSTERS = {
    "hpc": hpc_config.HPC_PARTITIONS,
    "param": hpc_config.PARAM_PARTITIONS,
}

for cluster, table in CLUSTERS.items():
    print(f"\n=== {cluster} ===")

    print("npn * cpt fills exactly one node (no idle cores, no oversubscription)")
    for name, p in table.items():
        if p.cores_per_node > p.cpu_cap:
            continue                # test: the cap, not the node, is the limit
        plan = hpc_config.plan_partition(name, partitions=table)
        check(f"{cluster}/{name}: {plan['npn']} x {plan['cpt']} == {p.cores_per_node} cores",
              plan["npn"] * plan["cpt"], p.cores_per_node)

    print("\none MPI rank per NUMA domain")
    for name, p in table.items():
        check(f"{cluster}/{name}: npn == {p.numa_per_node} NUMA domains",
              hpc_config.plan_partition(name, partitions=table)["npn"],
              p.numa_per_node)

    print("\nno plan exceeds its QoS cap (would sit pending on "
          "QOSMaxCpuPerUserLimit)")
    for name, p in table.items():
        for nodes in (1, 2, 3):
            plan = hpc_config.plan_partition(name, nodes=nodes, partitions=table)
            total = (plan["nodes"] * plan["npn"] * plan["cpt"]
                     * plan["max_concurrent_jobs"])
            check(f"{cluster}/{name} @ {nodes} node(s): {total} <= {p.cpu_cap} cores",
                  total <= p.cpu_cap, True)

print("\nhpc's test partition: 8-core QoS cap shrinks the layout instead of "
      "overrunning it")
plan = hpc_config.plan_partition("test")
check("test: 2 ranks x 4 threads", (plan["npn"], plan["cpt"]), (2, 4))
check("test: 1 concurrent job", plan["max_concurrent_jobs"], 1)

print("\nparam's debug partition: no QoS cap of its own, so the etiquette "
      "ceiling (1 node) is what limits concurrency, not a real ValueError-\n"
      "worthy overrun")
plan = hpc_config.plan_partition("debug", partitions=hpc_config.PARAM_PARTITIONS)
check("debug: 2 ranks x 24 threads", (plan["npn"], plan["cpt"]), (2, 24))
check("debug: 1 concurrent job", plan["max_concurrent_jobs"], 1)

print("\neach profiled host's literal defaults still equal "
      "plan_partition(host.partition) -- nodes auto-computed to fill the "
      "QoS cap (2026-08-23 resource-allocation policy) -- on its own table")
for host, table in ((hpc_config.HPC, hpc_config.HPC_PARTITIONS),
                     (hpc_config.PARAM, hpc_config.PARAM_PARTITIONS)):
    want = hpc_config.plan_partition(host.partition, partitions=table)
    got = {k: getattr(host, k) for k in want}
    check(f"{host.name} defaults", got, want)

print("\nhosts without a profiled table are not silently given another "
      "cluster's")
check("local: no partition table", hpc_config.LOCAL.partitions, {})

print("\nan unknown queue name is refused, not guessed at")
try:
    hpc_config.plan_partition("gpu")
except ValueError:
    print("  PASS  plan_partition('gpu') raises ValueError on hpc's table")
else:
    print("  FAIL  plan_partition('gpu') did not raise")
    failures.append("unknown partition (hpc)")

print("\nparam's gpu partition is deliberately excluded, not forgotten: it "
      "needs a --gres=gpu request plan_partition() has no field for, and\n"
      "this solver has no GPU code path")
try:
    hpc_config.plan_partition("gpu", partitions=hpc_config.PARAM_PARTITIONS)
except ValueError:
    print("  PASS  plan_partition('gpu', PARAM_PARTITIONS) raises ValueError")
else:
    print("  FAIL  plan_partition('gpu', PARAM_PARTITIONS) did not raise")
    failures.append("unknown partition (param gpu)")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
