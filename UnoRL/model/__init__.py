''' Register rule-based models or pre-trianed models
'''
from rlcard.models.registration import register, load

register(
    model_id = 'UNORuleModelSameNumber',
    entry_point = 'rlcard.models.uno_rule_models:UNORuleModelSameNumber')