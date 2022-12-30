# The new config inherits a base config to highlight the necessary modification
_base_ = 'spinenet_final_cycle.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=178),
        mask_head=dict(num_classes=178)))

# Modify dataset related settings
dataset_type = 'CocoDataset',
classes = ('Y-01', 'Y-02', 'Y-03', 'Y-04', 'Y-05', 'Y-06', 'Y-07', 'Y-08', 'Y-09', 'Y-10', 
'Y-11', 'Y-12', 'Y-13', 'Y-14', 'Y-15',	'Y-16',	'Y-17',	'Y-18',	'Y-19',	'Y-20',	
'Y-21',	'Y-22',	'Y-23',	'Y-24',	'Y-25',	'Y-26',	'Y-27',	'Y-28',	'Y-29',	'Y-30',	
'Y-31',	'Y-32',	'Y-33',	'Y-34',	'Y-35',	'Y-36',	'Y-37',	'Y-38',	'Y-39',	'Y-40',	
'Y-41',	'Y-42',	'Y-43',	'Y-44',	'Y-45',	'Y-46',	'Y-47',	'Y-48',	'Y-49',	'Y-50',	
'N-01',	'N-02',	'N-03',	'N-04',	'N-05',	'N-06',	'N-07',	'N-08',	'N-09',	'N-10',	
'N-11',	'N-12',	'N-13',	'N-14',	'N-15',	'N-16',	'N-17',	'N-18',	'N-19',	'N-20',	
'N-21',	'N-22',	'N-23',	'N-24',	'N-25',	'N-26',	'N-27',	'N-28',	'N-29',	'N-30',	
'N-31',	'N-32',	'N-33',	'N-34',	'N-35',	'N-36',	'N-37',	'N-38',	'N-39',	'N-40',	
'N-41',	'N-42',	'N-43',	'N-44',	'N-45',	'N-46',	'N-47',	'N-48',	'N-49',	'N-50',	
'C-01',	'C-02',	'C-03',	'C-04',	'C-05', 
'WO-01', 'WO-02', 'WO-03', 'WO-04', 'WO-05', 'WO-06', 'WO-07', 'WO-08', 'WO-09', 'WO-10', 
'WO-11', 'WO-12', 'WO-13', 'WO-14', 'WO-15', 'WO-16', 'WO-17', 'WO-18', 'WO-19', 'WO-20', 
'WO-21', 'WO-22', 'WO-23', 
'SO-01', 'SO-02', 'SO-03', 'SO-04', 'SO-05', 'SO-06', 'SO-07', 'SO-08', 'SO-09', 
'SO-11', 'SO-12', 'SO-13', 'SO-14', 'SO-15', 'SO-16', 'SO-17', 'SO-18', 'SO-19', 'SO-20', 
'SO-21', 'SO-22', 'SO-23', 'SO-24', 'SO-25', 'SO-26', 'SO-27', 'SO-28', 'SO-30', 
'SO-31', 'SO-32', 'SO-33', 'SO-34', 'SO-35', 'SO-36', 'SO-37', 'SO-38', 'SO-39', 'SO-40', 
'SO-41', 'SO-42', 'SO-43', 'SO-44', 'SO-45', 'SO-46', 'SO-47', 
'DO-01', 'DO-02', 'DO-03', 'DO-04', 'DO-06')
data = dict(
    train=dict(
        img_prefix='data/final_cycle/training/',
        classes=classes,
        ann_file='data/final_cycle/annotations/ContilTrain.json'),
    val=dict(
        img_prefix='data/final_cycle/validation/',
        classes=classes,
        ann_file='data/final_cycle/annotations/ContilValidation.json'),
    test=dict(
        img_prefix='data/final_cycle/test/',
        classes=classes,
        ann_file='data/final_cycle/annotations/ContilTest.json'))
   
    

    
