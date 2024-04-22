import os
import simulator
from importlib import reload
import sys
import json

TEST_DIRECTORY = './tests'


def main(student_algo: str):
    """
    Runs simulator and student algorithm on all tests in TEST_DIRECTORY
    Args:
        student_algo : Student algorithm to run
    """
    os.environ['model_name'] = 'hi_var' 
    EPOCHES = 1

    # Run main loop, print output
    
    print(f'\nTesting student algorithm {student_algo}')
    testfiles = [
            'lo_avg_lo_var.ini', 
            'mi_avg_lo_var.ini',
            'hi_avg_lo_var.ini',
            'lo_avg_mi_var.ini',
            'mi_avg_mi_var.ini',
            'hi_avg_mi_var.ini',
            'lo_avg_hi_var.ini', 
            'mi_avg_hi_var.ini',
            'hi_avg_hi_var.ini',
                    ]
    performance = {testfile[:-4] : {'QoE':0, 'Total Quality':0, 'Total Variation':0, 'Rebuffer Time':0} for testfile in testfiles}
    print(performance)
    for epoch in range(EPOCHES):
        sum_qoe = 0
        for test in testfiles:
            print(test)
            reload(simulator)
            quality, variation, rebuff, qoe = simulator.main(os.path.join(TEST_DIRECTORY, test), student_algo, False, False)
            performance[test[:-4]]['QoE'] = qoe
            performance[test[:-4]]['Total Quality'] = quality
            performance[test[:-4]]['Total Variation'] = variation
            performance[test[:-4]]['Rebuffer Time'] = rebuff
            print(f'\tTest {test: <12}:'
                  f' Total Quality {quality:8.2f},'
                  f' Total Variation {variation:8.2f},'
                  f' Rebuffer Time {rebuff:8.2f},'
                  f' Total QoE {qoe:8.2f}')
            sum_qoe += qoe

        print(f'\n\tEPOCH {epoch} -> Average QoE over all tests: {sum_qoe / len(testfiles):.2f}')
        performance['overall'] = sum_qoe / len(testfiles)

    with open(f"data/algo{sys.argv[1]}.json", "w") as f:
        json.dump(performance, f, indent=4)


if __name__ == "__main__":
    assert len(sys.argv) >= 2, f'Proper usage: python3 {sys.argv[0]} [student_algo]'
    if sys.argv[1] != 'RUN_ALL':
        main(sys.argv[1])
    else:
        for algo in os.listdir('./student'):
            if algo[:len('student')] != 'student':
                continue
            name = algo[len('student'):].split('.')[0]
            main(name)