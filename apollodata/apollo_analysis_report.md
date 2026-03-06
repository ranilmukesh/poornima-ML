# Dataset Quality & Statistical Report

**Source File:** /content/apolloCombined.csv
**Total Rows:** 238
**Total Columns:** 45

| Column                   | Type    | Missing (%)   |   Unique Count | Min   | Max   | Range   | Mean   | Median   | Unique Values Sample                                          |
|:-------------------------|:--------|:--------------|---------------:|:------|:------|:--------|:-------|:---------|:--------------------------------------------------------------|
| PostBLHBA1C              | float64 | 2.94%         |             65 | 4.0   | 13.1  | 9.1     | 7.24   | 6.6      | 65 unique values                                              |
| PostRgroupname           | int64   | 0.00%         |              2 | 1     | 2     | 1       | 1.47   | 1.0      | [2, 1]                                                        |
| PreRgender               | object  | 0.00%         |              2 | -     | -     | -       | -      | -        | ['M', 'F']                                                    |
| PreRarea                 | int64   | 0.00%         |              1 | 2     | 2     | 0       | 2.0    | 2.0      | [2]                                                           |
| PreRmaritalstatus        | float64 | 0.42%         |              4 | 1.0   | 5.0   | 4.0     | 1.49   | 1.0      | [1.0, 4.0, 2.0, 5.0, nan]                                     |
| PreReducation            | float64 | 2.10%         |              7 | 1.0   | 7.0   | 6.0     | 2.41   | 2.0      | [3.0, 4.0, 7.0, 2.0, 5.0, nan, 1.0, 6.0]                      |
| PreRpresentoccupation    | float64 | 0.42%         |              9 | 1.0   | 9.0   | 8.0     | 5.52   | 5.0      | [3.0, 4.0, 5.0, 9.0, 6.0, 8.0, nan, 2.0, 1.0, 7.0]            |
| PreRcurrentworking       | float64 | 1.26%         |              2 | 0.0   | 1.0   | 1.0     | 0.34   | 0.0      | [0.0, 1.0, nan]                                               |
| PreRdiafather            | float64 | 2.52%         |              2 | 0.0   | 1.0   | 1.0     | 0.12   | 0.0      | [0.0, nan, 1.0]                                               |
| PreRdiamother            | float64 | 4.20%         |              2 | 0.0   | 1.0   | 1.0     | 0.24   | 0.0      | [0.0, 1.0, nan]                                               |
| PreRdiabrother           | float64 | 5.46%         |              2 | 0.0   | 1.0   | 1.0     | 0.21   | 0.0      | [0.0, 1.0, nan]                                               |
| PreRdiasister            | float64 | 6.30%         |              2 | 0.0   | 1.0   | 1.0     | 0.12   | 0.0      | [0.0, 1.0, nan]                                               |
| PreRsleepquality         | float64 | 0.84%         |              4 | 1.0   | 4.0   | 3.0     | 1.62   | 1.0      | [1.0, 2.0, nan, 3.0, 4.0]                                     |
| PreRstworkvalue          | float64 | 1.26%         |             11 | 0.0   | 10.0  | 10.0    | 4.05   | 4.0      | [6.0, 7.0, 10.0, 3.0, 4.0, 1.0, 0.0, 5.0, 8.0, nan, 9.0, 2.0] |
| PreRstfamilyvalue        | float64 | 2.94%         |             11 | 0.0   | 10.0  | 10.0    | 4.28   | 4.0      | [0.0, 6.0, 7.0, 4.0, 3.0, 5.0, 2.0, 8.0, 10.0, nan, 1.0, 9.0] |
| PreRsthealthvalue        | float64 | 1.68%         |             11 | 0.0   | 10.0  | 10.0    | 5.48   | 5.0      | [7.0, 9.0, 4.0, 2.0, 5.0, 8.0, 0.0, 10.0, 3.0, 6.0, nan, 1.0] |
| PreRstfinancialvalue     | float64 | 2.94%         |             11 | 0.0   | 10.0  | 10.0    | 5.76   | 6.0      | [8.0, 9.0, 0.0, 5.0, 6.0, 7.0, 10.0, 2.0, 4.0, 3.0, nan, 1.0] |
| PreRmildactivityduration | float64 | 0.84%         |              6 | 0.0   | 5.0   | 5.0     | 2.24   | 2.0      | [2.0, 1.0, 5.0, 4.0, 3.0, 0.0, nan]                           |
| PreRmoderateduration     | float64 | 0.00%         |              6 | 0.0   | 5.0   | 5.0     | 2.26   | 2.0      | [3.0, 2.0, 5.0, 4.0, 0.0, 1.0]                                |
| PreRvigorousduration     | float64 | 2.52%         |              6 | 0.0   | 5.0   | 5.0     | 1.74   | 2.0      | [1.0, 3.0, 5.0, 4.0, 0.0, 2.0, nan]                           |
| PreRskipbreakfast        | float64 | 0.00%         |              3 | 1.0   | 3.0   | 2.0     | 2.36   | 2.0      | [3.0, 2.0, 1.0]                                               |
| PreRlessfiber            | float64 | 0.42%         |              3 | 1.0   | 3.0   | 2.0     | 1.89   | 2.0      | [2.0, 1.0, 3.0, nan]                                          |
| PreRlessfruit            | float64 | 0.42%         |              3 | 1.0   | 3.0   | 2.0     | 1.91   | 2.0      | [2.0, 1.0, 3.0, nan]                                          |
| PreRlessvegetable        | float64 | 0.00%         |              3 | 1.0   | 3.0   | 2.0     | 1.52   | 1.0      | [2.0, 3.0, 1.0]                                               |
| PreRmilk                 | float64 | 0.00%         |              3 | 1.0   | 3.0   | 2.0     | 1.76   | 2.0      | [2.0, 1.0, 3.0]                                               |
| PreRmeat                 | float64 | 0.42%         |              3 | 1.0   | 3.0   | 2.0     | 1.79   | 2.0      | [2.0, 1.0, 3.0, nan]                                          |
| PreRfriedfood            | float64 | 0.42%         |              3 | 1.0   | 3.0   | 2.0     | 2.2    | 2.0      | [2.0, 1.0, 3.0, nan]                                          |
| PreRsweet                | float64 | 0.42%         |              3 | 1.0   | 3.0   | 2.0     | 2.69   | 3.0      | [3.0, 2.0, 1.0, nan]                                          |
| PreRdrink                | float64 | 0.84%         |              3 | 1.0   | 3.0   | 2.0     | 2.58   | 3.0      | [2.0, 3.0, 1.0, nan]                                          |
| PreRstaplefood           | object  | 0.00%         |              7 | -     | -     | -       | -      | -        | ['1,3', '1,2,3', '1', '2', '1,2', '3', '5']                   |
| PreRweight               | float64 | 0.00%         |             69 | 34.0  | 110.0 | 76.0    | 67.28  | 68.0     | 69 unique values                                              |
| PreRhip                  | float64 | 41.18%        |             34 | 62.0  | 138.0 | 76.0    | 99.49  | 98.0     | 34 unique values                                              |
| PreRwaist                | float64 | 41.18%        |             34 | 70.0  | 130.0 | 60.0    | 91.78  | 90.0     | 34 unique values                                              |
| PreBLPPBS                | int64   | 0.00%         |            108 | 91    | 384   | 293     | 169.76 | 155.0    | 108 unique values                                             |
| PreBLFBS                 | int64   | 0.00%         |             75 | 61    | 248   | 187     | 100.32 | 92.0     | 75 unique values                                              |
| PreBLHBA1C               | float64 | 0.00%         |             64 | 3.3   | 17.4  | 14.1    | 6.97   | 6.55     | 64 unique values                                              |
| PreBLCHOLESTEROL         | int64   | 0.00%         |             94 | 106   | 298   | 192     | 178.74 | 175.0    | 94 unique values                                              |
| PreBLTRIGLYCERIDES       | int64   | 0.00%         |            103 | 75    | 549   | 474     | 150.13 | 139.0    | 103 unique values                                             |
| PreRsystolicfirst        | int64   | 0.00%         |             39 | 90    | 188   | 98      | 131.11 | 130.0    | 39 unique values                                              |
| PreRdiastolicfirst       | float64 | 0.42%         |             32 | 60.0  | 117.0 | 57.0    | 83.64  | 82.0     | 32 unique values                                              |
| postblage                | int64   | 0.00%         |             50 | 28    | 6262  | 6234    | 81.79  | 56.0     | 50 unique values                                              |
| Diabetic_Duration        | float64 | 0.00%         |             23 | 0.0   | 40.0  | 40.0    | 3.67   | 0.0      | 23 unique values                                              |
| Duration_Status          | float64 | 1.26%         |              2 | 0.0   | 1.0   | 1.0     | 0.42   | 0.0      | [0.0, 1.0, nan]                                               |
| current_smoking          | float64 | 5.04%         |              2 | 0.0   | 1.0   | 1.0     | 0.04   | 0.0      | [0.0, 1.0, nan]                                               |
| current_alcohol          | float64 | 6.30%         |              2 | 0.0   | 1.0   | 1.0     | 0.05   | 0.0      | [0.0, nan, 1.0]                                               |