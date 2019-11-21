#import os
#for module in os.listdir(os.path.dirname(__file__)):
#    if module == '__init__.py' or module[-3:] != '.py':
#        continue
#    __import__(module[:-3], locals(), globals())
#del module
#
#all_modules = os.listdir(os.path.dirname(__file__))
#print(all_modules)
#__all__ = all_modules

#need go check the installation of specific models that are required



__all__ = [ 'f_toolbox', 'f_trials', 'f_choices', 'f_psychometrics' ,\
           'f_probabilityDistortion', 'f_sql', 'f_reference', 'f_utility', \
           'f_Rfunctions', 'f_uncertainty', 'f_models', 'f_random', 'f_adaptive']
