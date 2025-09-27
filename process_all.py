#!/usr/bin/env python3
"""
Complete Diabetes Dataset Processing Pipeline
===========================================

This script runs the complete processing pipeline:
1. Process raw data and select important features
2. Apply optimal imputation to fill missing values
3. Generate final ML-ready datasets

Usage:
    python process_all.py
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    print("=" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}:")
        print(f"Command: {command}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'), 
        ('scikit-learn', 'sklearn')
    ]
    missing_packages = []
    
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            missing_packages.append(display_name)
            print(f"  ❌ {display_name}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install pandas numpy scikit-learn")
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def main():
    """Run the complete processing pipeline"""
    
    print("🚀 DIABETES DATASET PROCESSING PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 1: Process raw data
    success1 = run_command("python columns.py", "Data preprocessing and feature selection")
    
    if not success1:
        print("❌ Pipeline failed at preprocessing step")
        sys.exit(1)
    
    # Step 2: Apply optimal imputation  
    success2 = run_command("python final_imputation.py", "Optimal imputation")
    
    if not success2:
        print("❌ Pipeline failed at imputation step")
        sys.exit(1)
    
    # Step 3: Run efficiency check (optional)
    print("\n🔍 Running efficiency evaluation...")
    success3 = run_command("python simple_efficiency_check.py", "Efficiency evaluation")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"⏱️  Total processing time: {total_time:.1f} seconds")
    print(f"📁 Final datasets available in: final_imputed_data/")
    print(f"📊 Datasets processed:")
    
    # Check output files
    output_dir = "final_imputed_data"
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.csv'):
                filepath = os.path.join(output_dir, file)
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"   - {file} ({file_size:.1f} MB)")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Use datasets in final_imputed_data/ for machine learning")
    print(f"   2. All datasets have zero missing values")  
    print(f"   3. All categorical variables are numerically encoded")
    print(f"   4. Ready for scikit-learn, pandas, or any ML library")

if __name__ == "__main__":
    main()