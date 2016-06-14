%generate hdf5 locations test/train

% f = fopen('./moth_example/experiments/architectures/train_file_location_t_200k_rand_epoch.txt','w');
f = fopen('./moth_example/experiments/training_samples/train_file_location_t_1k.txt','w');


randomize_epochs = false;

total_iterations = 10000;
batch_size = 32;
total_samples = batch_size * total_iterations;

hdf5_size = 1000; %number of samples in a single hdf5 file

if(~ randomize_epochs)

    num_hdf5_batches = 1;

    for i=1:1:num_hdf5_batches
       if( i < num_hdf5_batches)
            fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/t/moth_train_',num2str(i),'.hdf5\n']); 
       else
            fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/t/moth_train_',num2str(i),'.hdf5']);
       end
    end
else %want to shuffle hdf5 files across multiple epochs
    num_hdf5_batches_orig = 50;

    for i=1:1:num_hdf5_batches_orig
            fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/t/moth_train_',num2str(i),'.hdf5\n']); 
    end
    
    num_hdf5_batches_additional = ceil((total_samples - (num_hdf5_batches_orig*hdf5_size))/hdf5_size);
    
    random_hdf5_idx = 1:1:num_hdf5_batches_additional; %need this many additional hdf5 files to ensure each epoch sees data in different order.
    random_hdf5_idx = mod(random_hdf5_idx-1,num_hdf5_batches_orig)+1;
    
    random_idx = randperm(length(random_hdf5_idx));
    random_hdf5_idx = random_hdf5_idx(random_idx);
    
    for i=1:1:length(random_hdf5_idx)
        if(i < length(random_hdf5_idx))
            fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/t/moth_train_',num2str(random_hdf5_idx(i)),'.hdf5\n']);
        else
            fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/t/moth_train_',num2str(random_hdf5_idx(i)),'.hdf5']);
        end
    end
    
    
end

fclose(f);

%commented out so we don't modify it for now
% f = fopen('./moth_example/test_file_location.txt','w');
% fprintf(f,['/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/moth_test_',num2str(1),'.hdf5']); 
% fclose(f);


