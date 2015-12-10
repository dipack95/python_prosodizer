import os
targetPath = os.path.abspath("../csv")
for path, dirs, files in os.walk(targetPath):
	for name in files:
		if name.endswith((".csv")):
			print("Path: " + str(os.path.join(path, name)))