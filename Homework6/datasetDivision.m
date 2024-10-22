function [new_datasets] = datasetDivision(Input, numClasses, numFolders)
    m = length(Input);
    for i = 1:numFolders
        
        numRowsPerClass = m/numClasses;
        numRowsPerFolder = numRowsPerClass/numFolders;
        new_datasets = cell(1,numFolders);
        for i=1:numFolders
            % Calculate the starting and ending indices for each original dataset
            start_idx = (i - 1) * numRowsPerFolder + 1;
            end_idx = i * numRowsPerFolder;
            %% Extract the rows from each original dataset and concatenate
            new_datasets{i} = [Input(start_idx:end_idx,:)];
            for k=1:(numClasses-1)
                start_idx = start_idx + numRowsPerClass;
                end_idx = end_idx + numRowsPerClass;
                new_datasets{i} = [new_datasets{i};Input(start_idx:end_idx,:)];
            end
        end     
    end
end