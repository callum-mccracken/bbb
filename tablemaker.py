import tools

# Todd's categories

#22.1    6.9
#3.22    
#22.5    47.6

# that's 46% left column, 54% right
# so say we have N events
N = 10000
l_factor = 0.46
r_factor = 0.54

# numbers of events in each category
tools.table_plot(22.1/100*N/l_factor,  # good 4th pick
                 3.22/100*N/l_factor,  # bad 4th pick
                 22.5/100*N/l_factor,  # bad ignore
                 6.90/100*N/r_factor,  # should have ignored
                 47.6/100*N/r_factor,  # good ignore
                 savename='todd_compare')