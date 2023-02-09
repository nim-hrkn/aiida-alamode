for x in 0001;
do echo $x
    (cd $x; bash ../run.sh; cp free_energy_history.txt ..)
done
for x in 0002 0003 0004 0005 0006 0007 0008 0009;
do echo $x
    (cd $x; cp ../free_energy_history.txt .;  bash ../run.sh; cp free_energy_history.txt ..)
done

