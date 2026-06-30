import json, os, objaverse

furn = json.load(open("gobjaverse_280k_Furnitures.json"))

# OPTIONAL: keep only the 351 you already rendered
root = os.path.expanduser("~/datasets")
furn = [e for e in furn if os.path.isdir(os.path.join(root, e))]

idx2obj = json.load(open("gobjaverse_280k_index_to_objaverse.json"))

uids = []
for e in furn:
    if e not in idx2obj:          # not every render index is in the map
        continue
    val = idx2obj[e]              # ".../<uid>.glb"  (or a bare uid)
    uids.append(os.path.splitext(os.path.basename(val))[0])

uids = sorted(set(uids))
print(f"resolving {len(uids)} GLBs")

# downloads to ~/.objaverse/hf-objaverse-v1/glbs/.../<uid>.glb
paths = objaverse.load_objects(uids, download_processes=8)   # {uid: local_glb_path}
json.dump(paths, open("furniture_glb_paths.json", "w"))