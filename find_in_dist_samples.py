from ase.db import connect
from collections import Counter
import os

db = connect('/root/autodl-tmp/2d-defect-dast/imp2d.db')

setups_dir = '/root/autodl-tmp/2d-defect-dast/gpaw-setups'
gpaw_elements = set([f.split('.')[0] for f in os.listdir(setups_dir) if f.endswith('.PBE.gz')])

hosts = Counter()
dopants = Counter()
combos = set()

for row in db.select():
    hosts[row.host] += 1
    dopants[row.dopant] += 1
    combos.add((row.host, row.dopant))

top_hosts = [h[0] for h in hosts.most_common(20)]
top_dopants = [d[0] for d in dopants.most_common(50) if d[0] in gpaw_elements]

print("Searching for missing combinations...")
found = 0
candidates = []
for host in top_hosts:
    for dopant in top_dopants:
        if (host, dopant) not in combos:
            candidates.append((host, dopant))

print(f"Found {len(candidates)} missing combinations.")
print("Some examples:", candidates[:20])

