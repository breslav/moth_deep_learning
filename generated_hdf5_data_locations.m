%generate hdf5 locations test/train

f = fopen('./moth_example/experiments/training_samples/train_file_location_t_20k.txt','w');

num_batches = 20;

for i=1:1:num_batches
   if( i < num_batches)
        fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/t/moth_train_',num2str(i),'.hdf5\n']); 
   else
        fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/t/moth_train_',num2str(i),'.hdf5']);
   end
end

fclose(f);

%commented out so we don't modify it for now
% f = fopen('./moth_example/test_file_location.txt','w');
% fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/moth_test_',num2str(1),'.hdf5']); 
% fclose(f);


