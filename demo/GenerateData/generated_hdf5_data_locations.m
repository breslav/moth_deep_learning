%create .txt file storing the location of already generated hdf5 files for both training and testing


%code for training
f = fopen('./moth_example/experiments/training_samples/train_file_location_t_20k.txt','w');

%number of hdf5 batches created
num_hdf5_batches = 20;

%here we write the paths of the hdf5 training files to a text file

for i=1:1:num_hdf5_batches
   if( i < num_hdf5_batches)
        fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/t/moth_train_',num2str(i),'.hdf5\n']); 
   else
        fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/t/moth_train_',num2str(i),'.hdf5']);
   end
end

fclose(f);


%code for testing hdf5 file (single)

f = fopen('./moth_example/test_file_location.txt','w');
fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/moth_test_',num2str(1),'.hdf5']); 
fclose(f);


