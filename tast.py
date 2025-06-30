# Test script
from runbook import RunbookFolderScanner

scanner = RunbookFolderScanner()  # Will use 7B by default now
knowledge = scanner.scan_runbook_folder("./runbooks/SHFEMR")
print(f"Success! Dataset: {knowledge.dataset_name}")