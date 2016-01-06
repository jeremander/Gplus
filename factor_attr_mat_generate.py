import subprocess

attr_types = ['employer', 'major', 'places_lived', 'school']
embeddings = ['adj', 'normlap', 'regnormlap']
ks = [50, 100, 200, 400, 800]
cs = [50]

for c in cs:
    for k in ks:
        for embedding in embeddings:
            for attr_type in attr_types:
                subcmd = "python3 factor_attr_mat.py -a %s -p NPMI1s -e %s -s -d 0.0 -k %d -c %d" % (attr_type, embedding, k, c)
                cmd = 'qsub -q all.q -l num_proc=1,mem_free=%dG,h_rt=120:00:00 -b Y -V -cwd -j yes -o . -N %s_%s_k%d_c%d "%s"' % (k // 16 + 1, attr_type, embedding, k, c, subcmd)
                print(cmd)
                subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)