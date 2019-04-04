file_num = 9;
motor_constant1_3 = -1/0.09368;
motor_constant4_6 = -1/0.09064;
hz = 100;
for i=1:file_num
    fileName = strcat('MonitoringDataLog',int2str(i),'.txt');
    writefileName = strcat('MonitoringDataLog',int2str(i),'.csv');
    RawData = load(fileName);
    ProcessData=[];
    % Command Torque
    for j=1:3
        ProcessData(:,j) = motor_constant1_3*RawData(:,45+j);
        ProcessData(:,j+3) = motor_constant4_6*RawData(:,48+j);
    end
    tmp_qdot = zeros(size(RawData,1),6);
    for j=1:6
        tmp_qdot(2:size(RawData,1)+1,j) = RawData(:,15+j); % qdot_pre
    end
    for j=1:6
        ProcessData(:,j+6) = RawData(:,3+j); % q
        ProcessData(:,j+12) = RawData(:,15+j); % qdot
        ProcessData(:,j+18) = (ProcessData(:,j+12)-tmp_qdot(1:size(tmp_qdot,1)-1,j))*hz; % qddot
        ProcessData(:,j+24) = RawData(:,9+j); % q_desired
        ProcessData(:,j+30) = RawData(:,21+j); % qdot_desired
        ProcessData(:,j+36) = RawData(:,33+j); % dynamic torque
    end
    ProcessData(:,43) = RawData(:,52);
    ProcessData(:,44) = 1-RawData(:,52);
    
    Training = ProcessData(randperm(fix(size(ProcessData,1)*1.0)),:);
    Validation = ProcessData(randperm(fix(size(ProcessData,1)*1.0)),:);
    Testing = ProcessData(randperm(fix(size(ProcessData,1)*1.0)),:);
    
    csvwrite(writefileName, ProcessData);
    %csvwrite(writefileName, Training);
    %csvwrite(writefileName, Validation);
    %csvwrite(writefileName, Testing);
end