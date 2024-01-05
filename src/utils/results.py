from ast import literal_eval

def read(nexp, method, path, env, 
    columns = ['it','reward','time','nrollouts','nsimulations'],
    time_constrained=False, max_time_minutes=10):

    results = []
    for exp in range(nexp):
        # preparing results dict
        results.append({})
        for column in columns:
            results[-1][column] = []

        # reading the data
        with open(path+method+'_'+env+'_'+str(exp)+'.csv','r') as resultfile:
            count, running_time = 0, 0.0
            for line in resultfile:
                if count > 0:
                    fcolumns = line.split(';')
                    for i in range(len(columns)):
                        results[exp][columns[i]].append(float(fcolumns[i]))
                    
                    if time_constrained:
                        running_time += float(fcolumns[2])
                        if running_time > max_time_minutes*60:
                            break
                count += 1
    return results

def read_estimation(nexp, method, estimation, path, env, 
    columns = ['it','reward','time','nrollouts','nsimulations',\
               'typeestimation','parameterestimation','memory'],
    time_constrained=False, max_time_minutes=10):

    results = []
    for exp in range(nexp):
        # preparing results dict
        results.append({})
        for column in columns:
            results[-1][column] = []

        # reading the data
        with open(path+method+'_'+estimation+'_'+env+'_'+str(exp)+'.csv','r') as resultfile:
            count, running_time = 0, 0.0
            for line in resultfile:
                if count > 0:
                    fcolumns = line.split(';')
                    for i in range(len(columns)):
                        if i > 4:
                            new_string = fcolumns[i]
                            if 'array' in fcolumns[i]:
                                for char in ['a','r','y','(',')']:
                                    new_string = new_string.replace(char,'')
                            results[exp][columns[i]].append(literal_eval(new_string))
                        else:
                            results[exp][columns[i]].append(float(fcolumns[i]))
                    
                    if time_constrained:
                        running_time += float(fcolumns[2])
                        if running_time > max_time_minutes*60:
                            break
                count += 1
    return results