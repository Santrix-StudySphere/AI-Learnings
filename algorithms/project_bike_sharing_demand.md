-----------------------------------------------------------------
 Bike Sharing Demand prediction Project for the hourly dataset
----------------------------------------------------------------------

# Conclusion or findings after the Data Visualization:

A. Contineous Features:
    1. Demand is not Normally distributed
    2. Temperature and demand appears to have a direct correlation.
    3. 'Temperature' and 'aTemp'(feel like Temerature) appear almost identical.
    4. Humidity and Windspeed affects demand but needs more statistical analysis.

B. Catagorical Features:

    1. There is a variation in demand based on:
        Season
        Month
        Holiday
        Hour
        Weather
    2. No significant change in demand due to workday or holiday (hence this can be excluded)
    3. Year wise growth pattern not considered dure to less number of years for analysis.


# Finding Outiers Analysis:
Range    Demand of bikes
0.05      5.00
0.10      9.00
0.15     16.00
0.90    451.20
0.95    563.10
0.99    782.22

It means :
5% of times the demand is between 0 to 5     -> Outlier
90% of times the demand is between 0 to 451
5% of times the demand is between 451 to 563
1% of time the demand is 782   -> Outlier


