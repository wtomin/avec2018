import csv
import matplotlib.mlab as mlab    
import matplotlib.pyplot as plt
import numpy as np 
import pdb  
def read_csv(file_path):
    instances = []
    subjects = []
    ages = []
    female = 0
    male = 0
    ymrs = []
    mania_level = []
    dev_n = 0
    train_n = 0
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            instances.append(row['Instance_name'])
            if not row['SubjectID'] in subjects:
                subjects.append(row['SubjectID'])
                ages.append(int(row['Age']))
            ymrs.append(int(row['Total_YMRS']))
            mania_level.append(int(row['ManiaLevel']))
            if row['Partition'] == 'dev':
                dev_n+=1
            elif row['Partition'] == 'train':
                train_n+=1
    # remove repetitive subjects
    subjects = list(set(subjects))
    for sub in subjects:
        if sub.startswith('MP'):
            male+=1
        else:
            female+=1
    n_samples = len(instances)
    return n_samples, subjects, ages, female, male, ymrs, np.asarray(mania_level), dev_n, train_n

def draw_pie(labels, data, title):
    
    plt.figure()
    if len(labels)==2:
        colors = ['#ff9999','#66b3ff']
        explode = (0.05, 0.05)
    elif len(labels)==3:
        colors = ['#ff9999','#66b3ff', '#99ff99']
        explode = (0.05, 0.05, 0.05)
    plt.pie(data, colors=colors, explode= explode, labels = labels, autopct='%2.1f%%',
            startangle=90, pctdistance=0.85)
    #draw circle
    centre_circle = plt.Circle((0,0),0.50,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')  
    #plt.tight_layout()
    plt.title(title)
    plt.savefig(title+'.png')
    plt.show()
    
    
def draw_hist(bins, data, title):
    plt.figure()
    a = np.array(data)
    plt.hist(a, bins)
    plt.title(title)
    plt.savefig(title+'.png')
    plt.show()
    
    
def main():
    csv_f = 'labels_metadata.csv'
    n_samples, subjects, ages, female, male, ymrs, mania_level, dev_n, train_n= read_csv(csv_f)
    print('In dataset: {} Subjects, {} videos'.format(len(subjects), n_samples))
    # data distribution
    labels = ['train','development']
    data = [train_n, dev_n]
    draw_pie(labels, data, 'Data_Distribution')
    # Gender Distribution
    labels = ['male', 'female']
    data = [male, female]
    draw_pie(labels, data,'Gender_Distribution')
    # age hist
    bins = np.linspace(min(ages), max(ages), 10, endpoint=True)
    data = ages
    draw_hist(bins, data, 'Ages_Histogram')
    #ymrs histogram
    bins = np.linspace(min(ymrs), max(ymrs), 10, endpoint=True)
    data = ymrs
    draw_hist(bins, data, 'YMRS_Histogram')
    #mania distribution
    labels = ['remission', 'hypomania','mania']
    data = [sum(mania_level==1), sum(mania_level==2), sum(mania_level==3)]
    draw_pie(labels, data, 'Mania_Level_Distribution')
    

if __name__ == "__main__":
    pdb.set_trace()
    main()