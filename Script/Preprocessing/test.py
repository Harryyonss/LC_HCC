import glob
import pydicom

files = glob.glob("/data/backed_up/shared/ME_7T_Pilot/Raw/11111/run-700/700-NOT_DIAGNOSTIC__MBME_solo_32ch/*")

echo_times = set()
echo_numbers = set()

for f in files[:500]:
  try:
    ds = pydicom.dcmread(f, stop_before_pixels=True)
    if "EchoTime" in ds:
      echo_times.add(float(ds.EchoTime))
    if "EchoNumbers" in ds:
      echo_numbers.add(ds.EchoNumbers)
  except Exception:
    pass

print(f"Echo Numbers: {sorted(list(echo_numbers))}")
print(f"Echo Times (TE): {sorted(list(echo_times))}")