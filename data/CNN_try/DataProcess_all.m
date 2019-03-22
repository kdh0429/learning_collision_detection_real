% Change DataType to 'Testing'/'Validation'/'Training'
DataType = 'Testing';

motor_constant1_3 = -1/0.09368;
motor_constant4_6 = -1/0.09064;
hz = 100;

num_input = 36;
num_time_step = 20;

fileName = strcat('../MonitoringDataLog',DataType,'.txt');
RawData = load(fileName);
ProcessData=zeros(size(RawData,1),num_input*num_time_step+2);


for k=num_time_step:size(RawData,1)
    ProcessData(k,num_input*num_time_step+1) = RawData(k,52);
    ProcessData(k,num_input*num_time_step+2) = 1-RawData(k,52);
    for i=1:num_time_step
        
        for j=1:3
            for l=1:6
                ProcessData(k,num_input*(i-1)+6*l) = motor_constant1_3*RawData(k-i+1,45+j);
            end
        end
        for j=1:3
            ProcessData(k,num_input*(i-1)+6*(j-1)+1) = motor_constant1_3*RawData(k-i+1,45+j);
            ProcessData(k,num_input*(i-1)+6*(j+2)+1) = motor_constant4_6*RawData(k-i+1,48+j);
        end
        for j=1:6
            ProcessData(k,num_input*(i-1)+6*(j-1)+2) = RawData(k-i+1,3+j); % q
            ProcessData(k,num_input*(i-1)+6*(j-1)+3) = RawData(k-i+1,15+j); % qdot
            ProcessData(k,num_input*(i-1)+6*(j-1)+4) = RawData(k-i+1,9+j); % q_desired
            ProcessData(k,num_input*(i-1)+6*(j-1)+5) = RawData(k-i+1,21+j); % qdot_desired
            ProcessData(k,num_input*(i-1)+6*(j-1)+6) = RawData(k-i+1,33+j); % dynamic torque
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

