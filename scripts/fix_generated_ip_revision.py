#!/usr/bin/env python3
"""Project-local workaround for AMD AR76960, not an installed-tool/clock modification."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('generated_tcl');p.add_argument('--generated-root',default='build/partial-f16-hw');a=p.parse_args();f=Path(a.generated_tcl).resolve()
repo=Path(__file__).resolve().parents[1];build=(repo/'build').resolve();root=Path(a.generated_root).resolve()
assert root!=build and root.is_relative_to(build), 'only a generated subproject inside this repository build directory is permitted'
assert f.is_relative_to(root) and f.name=='run_ippack.tcl'
s=f.read_text();match=re.search(r'^set Revision\s+"(\d+)"',s,re.M);assert match and int(match[1])>2147483647
backup=f.with_suffix('.tcl.original');assert not backup.exists();backup.write_text(s)
new=s[:match.start()]+re.sub(r'"\d+"','"1"',match[0])+s[match.end():];f.write_text(new)
report={'scope':'generated_project_IP_metadata_only','generated_root':str(root),'reference':'https://docs.amd.com/r/en-US/76960','old_revision':match[1],'new_revision':1,'original_sha256':hashlib.sha256(s.encode()).hexdigest(),'modified_sha256':hashlib.sha256(new.encode()).hexdigest(),'kernel_source_modified':False,'system_clock_modified':False,'installed_tools_modified':False}
f.with_name('revision-fix.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
