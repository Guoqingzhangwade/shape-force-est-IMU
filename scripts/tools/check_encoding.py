import os

project_path = r"C:\Users\Guoqing Zhang\OneDrive - stevens.edu\Documents\GitHub\shape-force-est-IMU"

for root, dirs, files in os.walk(project_path):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, encoding='utf-8') as f:
                    f.read()
            except UnicodeDecodeError as e:
                print(f"UnicodeDecodeError in: {file_path}")
                print(f"Error details: {e}\n")
