#!/usr/bin/env python3
"""
Calculate accuracy for BBH test results
"""
import json
import sys

def calculate_accuracy(file_path):
    """Calculate accuracy for each list in test_acc array"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_acc = data.get('test_acc', [])
        
        if not test_acc:
            print("test_acc data not found")
            return
        
        print(f"Total {len(test_acc)} test rounds")
        print("-" * 50)
        
        for i, acc_list in enumerate(test_acc):
            if not acc_list:
                print(f"Round {i+1}: No data")
                continue
                
            correct = sum(acc_list)
            total = len(acc_list)
            accuracy = correct / total * 100
            
            print(f"Round {i+1}: {correct}/{total} = {accuracy:.2f}%")
        
        # Calculate overall statistics
        if test_acc:
            all_results = [item for sublist in test_acc for item in sublist]
            total_correct = sum(all_results)
            total_tests = len(all_results)
            overall_accuracy = total_correct / total_tests * 100
            
            print("-" * 50)
            print(f"Overall result: {total_correct}/{total_tests} = {overall_accuracy:.2f}%")
            
            # Calculate average accuracy per round
            round_accuracies = []
            for acc_list in test_acc:
                if acc_list:
                    round_accuracies.append(sum(acc_list) / len(acc_list) * 100)
            
            if round_accuracies:
                avg_accuracy = sum(round_accuracies) / len(round_accuracies)
                print(f"Average round accuracy: {avg_accuracy:.2f}%")
                print(f"Highest round accuracy: {max(round_accuracies):.2f}%")
                print(f"Lowest round accuracy: {min(round_accuracies):.2f}%")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"JSON parsing error: {file_path}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Default file paths
    default_file_m = "./results/results_BBH_object_counting_gpt-4o_v1_momentum.json"
    default_file_v1 = "./results/results_BBH_object_counting_gpt-4o_v1.json"
    default_file_v2 = "./results/results_BBH_object_counting_gpt-4o_v2.json"
    
    # Use provided file path if command line argument is given
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file_v2
    
    print(f"Analyzing file: {file_path}")
    calculate_accuracy(file_path)