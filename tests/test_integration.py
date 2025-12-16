
import unittest
import subprocess
import os
import shutil
import h5py
import numpy as np

VERBOSE = True

class TestDatasetIntegration(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.test_data_dir = os.path.join(self.root_dir, "data", "test_integration_data")

        # Clean up before start
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def tearDown(self):
        # Clean up after test
        if os.path.exists(self.test_data_dir):
             shutil.rmtree(self.test_data_dir)
        pass

    def run_command(self, cmd_list):
        if VERBOSE:
            print(f"Running: {' '.join(cmd_list)}")
        result = subprocess.run(
            cmd_list,
            cwd=self.root_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
        self.assertEqual(result.returncode, 0, f"Command failed: {' '.join(cmd_list)}")

    def test_single_object_S(self):
        print("\n=== Test Case 1: Single Object S ===")
        dataset_name = "test_integration_data"

        # 1. Run Generation (1 cycle, few drops)
        cmd_gen = [
            "uv", "run", "1_pybullet_create_n_collect.py",
            "--start_cycle", "1",
            "--end_cycle", "1",
            "--mode", "direct",
            "--renderer", "tiny",
            "--dataset_name", dataset_name,
            "--model_name", "teris",
            "--object_types", "S",
            "--max_drop", "5",
            "--dropping", "falling"
        ]
        self.run_command(cmd_gen)

        # 2. Check CSV for verify class name "S"
        gt_csv = os.path.join(self.root_dir, "data", dataset_name, "training", "gt", "cycle_0001", "005.csv")
        self.assertTrue(os.path.exists(gt_csv), "GT CSV not generated")

        with open(gt_csv, 'r') as f:
            content = f.read()
            # Expect lines like: 0,S,0.1,0.2...
            self.assertIn(",S,", content, "CSV should contain class 'S'")
            self.assertNotIn(",I,", content, "CSV should NOT contain class 'I'")

        # 3. Run H5 Generation
        cmd_h5 = [
            "uv", "run", "4_generate_h5.py",
            "--dataset_name", dataset_name,
            "--model_name", "teris"
        ]
        self.run_command(cmd_h5)

        # 4. Check H5 for verify class ID 3 (S)
        h5_file = os.path.join(self.root_dir, "data", dataset_name, "training", "h5", "cycle_0001", "005.h5")
        self.assertTrue(os.path.exists(h5_file), "H5 file not generated")

        with h5py.File(h5_file, 'r') as f:
            labels = f['labels'][:]
            # Class ID is the last column (index 14)
            cls_ids = labels[:, 14]
            unique_ids = np.unique(cls_ids)
            print(f"Unique Class IDs in H5: {unique_ids}")

            # Should only contain 3.0
            self.assertTrue(np.all(unique_ids == 3.0), f"Expected only class ID 3 (S), found {unique_ids}")

    def test_multi_objects(self):
        print("\n=== Test Case 2: Multi Objects (I,S,O,T) ===")
        dataset_name = "test_integration_data"

        # 1. Run Generation
        cmd_gen = [
            "uv", "run", "1_pybullet_create_n_collect.py",
            "--start_cycle", "1",
            "--end_cycle", "1",
            "--mode", "direct",
            "--renderer", "tiny",
            "--dataset_name", dataset_name,
            "--model_name", "teris",
            "--object_types", "I", "S", "O", "T",
            "--max_drop", "20", # Enough to get variety
            "--dropping", "falling"
        ]
        self.run_command(cmd_gen)

        # 2. Run H5 Generation
        cmd_h5 = [
            "uv", "run", "4_generate_h5.py",
            "--dataset_name", dataset_name,
            "--model_name", "teris"
        ]
        self.run_command(cmd_h5)

        # 3. Check H5
        # Since random, we might not get ALL types in one scene, but we should see valid IDs.
        # Check standard map: I:0, O:1, T:2, S:3
        valid_ids = {0.0, 1.0, 2.0, 3.0}

        h5_file = os.path.join(self.root_dir, "data", dataset_name, "training", "h5", "cycle_0001", "020.h5")
        if not os.path.exists(h5_file):
             # Try another one if 20 doesn't exist? (max_drop 20 implies 20 files)
             h5_file = os.path.join(self.root_dir, "data", dataset_name, "training", "h5", "cycle_0001", "005.h5")

        with h5py.File(h5_file, 'r') as f:
            labels = f['labels'][:]
            cls_ids = labels[:, 14]
            unique_ids = np.unique(cls_ids)
            print(f"Unique Class IDs in Multi-Object H5: {unique_ids}")

            for uid in unique_ids:
                self.assertIn(uid, valid_ids, f"Found invalid class ID {uid}")

if __name__ == "__main__":
    unittest.main()
