# --- Demographic model ---


import msprime
import numpy as np
import tskit


def history_archaic(
    gen_time,
    len_seq,
    rr,
    mu,
    n_e,
    t,  # t включает t_SECOND_NEAND_MIGRATION
    n,
    rand_sd,
    n_neand,
    t_neand_samples,
    n_eu,
    n_eu_growth,
    t_eu_growth,
    n_eu_bottleneck,
    gr_rt,
    p_admix,
    p_admix2
):
    n_ANC, n_ND, n_AMH, n_OOA, n_AF, n_EU, n_Sample = n_e
    t_NEAND_migration, t_NEAND_AMH, t_OOF_AF, t_SECOND_NEAND_MIGRATION, t_ND_SPLIT, t_Split_EU_Sample = t
    
    demography = msprime.Demography()
    demography.add_population(name="AF", initial_size=n_AF)
    demography.add_population(name="EU", initial_size=n_EU)
    demography.add_population(name="AMH", initial_size=n_AMH)
    demography.add_population(name="NEAND", initial_size=n_ND)
    demography.add_population(name="ND1", initial_size=3400)
    demography.add_population(name="ND2", initial_size=3400)
    demography.add_population(name="ANCES", initial_size=n_ANC)  #common population for Neanderthal and AMH
    demography.add_population(name="OOA", initial_size = n_OOA)
    demography.add_population(name="OOA_admixed", initial_size=n_OOA)
    demography.add_population(name="SAMPLE", initial_size=n_Sample)
    demography.add_population(name="SAMPLE_admixed", initial_size=n_Sample)
    
    
    demography.add_population_parameters_change(time=0, initial_size=n_EU, population=1, growth_rate=gr_rt)
    demography.add_population_parameters_change(time=t_eu_growth, initial_size=n_eu_growth, population=1, growth_rate=0)
    
    demography.add_population_split(
        time = t_OOF_AF,
        derived = ["AF", "OOA"],
        ancestral = "AMH"
    )
    
    demography.add_population_split(
        time = t_NEAND_AMH,
        derived = ["AMH", "NEAND"],
        ancestral = "ANCES"
    )

    # 2. Разделение NEAND на ND1 и ND2
    demography.add_population_split(
        time = t_ND_SPLIT,
        derived = ["ND1", "ND2"],
        ancestral = "NEAND"
    )


    # 4. Примесь ND1 -> OOA (адмиксуем ND1 в OOA)
    #    proportions=[(1 - p_admix1), p_admix1]
    demography.add_admixture(
        time = t_NEAND_migration,
        derived = "OOA_admixed",
        ancestral = ["OOA", "ND1"],
        proportions = [1 - p_admix, p_admix]
    )

    # 5. Разделение OOA на EU и SAMPLE
    demography.add_population_split(
        time = t_Split_EU_Sample,
        derived = ["EU", "SAMPLE"],
        ancestral = "OOA_admixed"
    )

    # 6. Примесь ND2 -> SAMPLE
    demography.add_admixture(
        time = t_SECOND_NEAND_MIGRATION,
        derived = "SAMPLE_admixed",
        ancestral = ["SAMPLE", "ND2"],
        proportions = [1 - p_admix2, p_admix2]
    )

    demography.sort_events()
    print(demography.debug())

    ts = msprime.sim_ancestry(
        samples= [
            msprime.SampleSet(n_eu, ploidy=1, population='EU'),
            msprime.SampleSet(n, ploidy=1, population='AF'),
              msprime.SampleSet(n_Sample, ploidy=1, population="SAMPLE_admixed"),  # Sample from admixed population
            msprime.SampleSet(n_neand//2, ploidy=1, population="ND1", time=t_neand_samples),  # Split Neanderthal samples
            msprime.SampleSet(n_neand//2, ploidy=1, population="ND2", time=t_neand_samples)
        ],
        ploidy=1,
        sequence_length=len_seq,
        recombination_rate=rr,
        demography=demography,
        record_migrations=True
    )
   
    ts = msprime.sim_mutations(ts, rate=mu)
    return ts

#несколько вспомогательных функций
def connected(m):
    for i in range(len(m)-1):
        if m[i][1] == m[i+1][0]:
            return True
    return False
        
def remove_one(m):
    mas = m
    while connected(mas) == True:
        for i in range(len(mas)-1):
            if mas[i][1] == mas[i+1][0]:
                mas[i][1] = mas[i+1][1]
                mas.pop(i+1)
                break
    return mas

#def remove_one(m):
#    # Создадим копию и отсортируем интервалы по началу
#    mas = sorted(m, key=lambda interval: interval[0])
#    prev_len = len(mas)
#    while connected(mas):
#        for i in range(len(mas)-1):
#            if mas[i][1] == mas[i+1][0]:
#                mas[i][1] = mas[i+1][1]
#                mas.pop(i+1)
#                break
#        # Если после объединения длина не уменьшилась, прерываем цикл
#        if len(mas) == prev_len:
#            break
#        prev_len = len(mas)
#    return mas

#Вход: ts, название популяции, индивид(которого мы препарируем), время предка
def get_migrating_tracts_ind(ts, pop_name, ind, T_anc):
    
    pop=-1
    for i in ts.populations():
        if i.metadata['name']==pop_name:
            pop=i.id
    
    mig = ts.tables.migrations
    migration_int = []

    for tree in ts.trees():  #перебираем все деревья. Как известно, каждому дереву отвечает участок днк
        anc_node = ind #chose observable node
        while tree.time( tree.parent(anc_node) ) <= T_anc : #идем в прошлое до вершины anc_node по предкам нашего мексиканца, пока не наткнемся на миграцию
            anc_node = tree.parent(anc_node)
        migs = np.where(mig.node == anc_node)[0] #выбирем все строки, соответствующие заданному узлу

        #идем по таблице миграций с anc_node и проверяем, чтобы миграции попадали в тот самый участок днк
        for i in migs:

            stroka = mig[i]
            if stroka.time == T_anc and stroka.dest == pop and tree.interval.left >= stroka.left and tree.interval.right <= stroka.right:
                migration_int.append([tree.interval.left, tree.interval.right])

    migration_int2 = []
    for i in range(len(migration_int)):
        if migration_int[i][0] != migration_int[i][1]:
            migration_int2.append(migration_int[i])
    migration_int = migration_int2
    
    mi = remove_one(migration_int)
    mi.sort()

    return mi


# return European tracts with input=Neanderthal tracts
def tracts_eu(tr_nd, seq_length):
    result = []

    if tr_nd[0][0] > 0:
        result.append([0,tr_nd[0][0]-1])
        
    for i in range(len(tr_nd)-1):
        result.append([tr_nd[i][1]+1, tr_nd[i+1][0]-1])
        
    if tr_nd[-1][1]!=seq_length-1:
        result.append([tr_nd[-1][1]+1,seq_length-1])
      
    return result



##from Skov
def print_neand_dosages(ts):

    seq_len = ts.get_sequence_length()

    ME_ids = ts.get_samples(1)
    de_seg = {i: [] for i in ME_ids}
    ar_seg = {i: [] for i in ME_ids}
    for mr in ts.migrations():
        if mr.source == 1 and mr.dest == 3:
            for tree in ts.trees(leaf_lists=True):
                if mr.left > tree.get_interval()[0]:
                    continue
                if mr.right <= tree.get_interval()[0]:
                    break
                for l in tree.leaves(mr.node):
                    if l in ME_ids:
                        # print(l)
                        de_seg[l].append(tree.get_interval())

    def combine_segs(segs, get_segs = False):
        merged = np.empty([0, 2])
        if len(segs) == 0:
            if get_segs:
                return([])
            else:
                return(0)
        sorted_segs = segs[np.argsort(segs[:, 0]), :]
        for higher in sorted_segs:
            if len(merged) == 0:
                merged = np.vstack([merged, higher])
            else:
                lower = merged[-1, :]
                if higher[0] <= lower[1]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1, :] = (lower[0], upper_bound)
                else:
                    merged = np.vstack([merged, higher])
        if get_segs:
            return(merged)
        else:
            return(np.sum(merged[:, 1] - merged[:, 0])/seq_len)

    true_de_prop = [combine_segs(np.array(de_seg[i])) for i in sorted(de_seg.keys())]
    true_de_segs = [combine_segs(np.array(de_seg[i]), True) for i in sorted(de_seg.keys())]
    print("Neand ancestry: ", true_de_prop)
      
