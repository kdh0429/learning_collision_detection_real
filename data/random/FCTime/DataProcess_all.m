% Change DataType to 'Testing'/'Validation'/'Training'
DataType = 'Testing';

motor_constant1_3 = -1/0.09368;
motor_constant4_6 = -1/0.09064;
hz = 100;

num_input = 36;
num_time_step = 10;

fileName = strcat('../MonitoringDataLog',DataType,'.txt');
RawData = load(fileName);
ProcessData=zeros(size(RawData,1),num_input*num_time_step+2);


for k=num_time_step:size(RawData,1)
    ProcessData(k,num_input*num_time_step+1) = RawData(k,58);
    ProcessData(k,num_input*num_time_step+2) = 1-RawData(k,58);
    for i=1:num_time_step
        for j=1:3
            ProcessData(k,num_input*(i-1)+j) = motor_constant1_3*RawData(k-i+1,51+j);
            ProcessData(k,num_input*(i-1)+j+3) = motor_constant4_6*RawData(k-i+1,54+j);
        end
        for j=1:6
            ProcessData(k,num_input*(i-1)+j+6) = RawData(k-i+1,3+j); % q
            ProcessData(k,num_input*(i-1)+j+12) = RawData(k-i+1,15+j); % qdot
            ProcessData(k,num_input*(i-1)+j+18) = RawData(k-i+1,9+j); % q_desired
            ProcessData(k,num_input*(i-1)+j+24) = RawData(k-i+1,21+j); % qdot_desired
            ProcessData(k,num_input*(i-1)+j+30) = RawData(k-i+1,39+j); % dynamic torque
        end
    end
end



Training = ProcessData(randperm(fix(size(ProcessData,1)*1.0)),:);
Validation = ProcessData(randperm(fix(size(ProcessData,1)*1.0)),:);
Testing = ProcessData(randperm(fix(size(ProcessData,1)*1.0)),:);

csvwrite(strcat(DataType,'_raw_data_.csv'), ProcessData);

if strcmp(DataType,'Training')
    csvwrite('training_data_.csv', Training);
elseif strcmp(DataType,'Validation')
    csvwrite('validation_data_.csv', Validation);
elseif strcmp(DataType,'Testing')
    csvwrite('testing_data_.csv', Testing);
end





