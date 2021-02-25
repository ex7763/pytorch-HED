import os,pdb
import time


if __name__ == '__main__':
    #######################################################################################
    ## relative to config
    config_file='standard.yaml'
    #######################################################################################

    ckpt_dir = '../ckpt'
    main_dirs = [ckpt_dir]

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


    #######################################################################################

    time.ctime()
    cur_time = time.strftime('_%b%d_%H-%M-%S') 

    #cur_time = "_Feb24_12-34-59"
    cur_time = "_Feb25_00-27-19"

    ### run script
    cmd = '''\
        LOG="''' + ckpt_dir + '/' + '/'.join(dirs) + '/' + filename  + '''-`date +'%Y-%m-%d_%H-%M-%S'`_test";
        echo $LOG ;
        python run.py --mode test --cfg ''' +  config_file + ''' --time ''' + cur_time  + '''$2>&1 | tee ${LOG}
    '''

    os.system(cmd)
