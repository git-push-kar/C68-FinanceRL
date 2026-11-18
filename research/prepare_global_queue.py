"""Create the isolated full-history CPU queue from the audited serial driver."""
from pathlib import Path
root=Path(__file__).parent;out=root/'run_exposure_global_numeric_queue.py'
if out.exists():raise FileExistsError(out)
s=(root/'run_exposure_direct_numeric_queue.py').read_text()
s=s.replace('exposure_direct','exposure_global').replace('exposure_soft_numeric','exposure_direct_numeric')
s=s.replace('_256','_64').replace('budget256','budget64').replace("'256'","'64'").replace("'16'","'4'")
s=s.replace('CPU direct-reinforcement feasibility','CPU full-history direct-reinforcement feasibility')
s=s.replace('CPU direct RL feasibility only','CPU full-history direct RL feasibility only')
out.write_text(s);print(out)
