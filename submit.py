import sys
import os,pdb,time

#######################################################################################
## relative to config
config_file='no_norm.yaml'
#config_file='standard.yaml'
#cluster=True
#cluster=False
#######################################################################################


if __name__ == '__main__':
    ckpt_dir = '../ckpt'
    main_dirs = [ckpt_dir]

    ### Get job name
    with open('config/'+config_file,'r') as f:
        lines = f.readlines()
        job_name = lines[0][:-1].split(': ')[1][1:-1]

    ### process config
    config=config_file.split('/')
    print(config)
    if len(config)==1:
        filename = config[0]
        dirs =[]
        dirs.append(filename)
        dirs.append('log')
    else:
        dirs,filename = config[0:-1], config[-1]
        dirs.append(filename)
        dirs.append('log')
    print(dirs, filename)
    filename = filename.split('.')[0]

    for each_main_dir in main_dirs:
        for ind, each_dir in enumerate(dirs):
            each_dir = '/'.join( dirs[0:ind+1])
            new_dir = os.path.join(each_main_dir, each_dir) 
            if not os.path.exists( new_dir ):
                os.mkdir( new_dir )
        print('create ckpt dir: ', each_main_dir, '/',  each_dir)



    time.ctime()
    cur_time = time.strftime('_%b%d_%H-%M-%S') 
    layers = 3

    #######################################################################################


    ### run script
    cmd = '''\
        LOG="''' + ckpt_dir + '/' + '/'.join(dirs) + '/' + filename  + '''-`date +'%Y-%m-%d_%H-%M-%S'`_train" 
        echo $LOG ;
    '''
    cmd2 = f'python run.py --mode train --layers {layers} --cfg {config_file} --time {cur_time} $2>&1 | tee ${{LOG}}'
    print(cmd2)

    os.system(cmd + cmd2)
