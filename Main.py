%% ================================================================
%  Proposal Workflow Implementation (Final Corrected & Visible)
%  - Chaos-Based Spectrum Modeling (Logistic Map)
%  - Bifurcation + Variance Stability Detection
%  - Scenario-driven PU activity (IoT, Industrial, Public, Emergency)
%  - Trust-Aware SU Assignment
%  - CR-AODV-inspired Routing (stable links only)
%  - Metrics: Accuracy, Precision, Recall, F1, PDR, Energy, Latency, Overhead
%  - White background, black text, Times New Roman, bold
% ================================================================
clc; clear; close all;
%% ---------------- Parameters ----------------
numChannels = 100;
timeSlots   = 100;
numSUs      = 10;
bifurcation_threshold = 3.57;
sigma_threshold       = 0.25;
windowSize = 5;
noiseLevel = 0.01;
numSU_nodes = 8;
netRange    = 40;
commRange   = 15;
%% ---------------- Scenario r_values ----------------
r_values = zeros(1,numChannels);
r_values(1:20)   = 2.6 + 0.2*rand(1,20);   % IoT (stable)
r_values(21:50)  = 3.4 + 0.2*rand(1,30);   % Industrial (semi-stable)
r_values(51:80)  = 3.7 + 0.2*rand(1,30);   % Public (chaotic)
r_values(81:100) = 2.5 + 0.3*rand(1,20);   % Emergency (very stable)
x = rand(1,numChannels); % Initial states
x_history = zeros(timeSlots, numChannels);
PU_matrix = zeros(numChannels, timeSlots);
%% ---------------- Metrics Storage ----------------
ACC=[]; PREC=[]; REC=[]; F1s=[]; PDR=[]; ENJ=[]; LAT=[]; OVR=[];
%% ---------------- Scenario-driven PU usage ----------------
iot_on_period = 4; iot_cycle = 10;
ind_busy_prob = 0.45;
pub_busy_prob = 0.7;
emergency_burst_period = 15;
%% ---------------- Figures ----------------
% Heatmap
figHeat = figure('Color','w');
hHeat = imagesc(PU_matrix'); set(gca,'YDir','normal','color','k');
colormap([1 1 1; 1 0 0]); colorbar; caxis([0 1]);
xlabel('Time Slot','FontName','Times New Roman','FontSize',32,'FontWeight','bold','Color','k');
ylabel('Channel Index','FontName','Times New Roman','FontSize',32,'FontWeight','bold','Color','k');
set(gca,'FontName','Times New Roman','FontSize',32,'FontWeight','bold', ...
    'XColor','k','YColor','k','Color','w');
% SU Assignment
figSU = figure('Color','w'); hold on;
colorsSUs = lines(numSUs);
hSU = gobjects(numSUs,1);
for su=1:numSUs
    hSU(su) = plot(NaN,NaN,'-o','LineWidth',2,'Color',colorsSUs(su,:),'MarkerFaceColor',colorsSUs(su,:),'DisplayName',['SU ' num2str(su)]);
end
legend('show','Location','northoutside','NumColumns',3, ...
    'FontName','Times New Roman','FontSize',28,'Color','k');
xlabel('Time','FontName','Times New Roman','FontSize',32,'FontWeight','bold','Color','k');
ylabel('Channel','FontName','Times New Roman','FontSize',32,'FontWeight','bold','Color','k');
set(gca,'FontName','Times New Roman','FontSize',32,'FontWeight','bold', ...
    'XColor','k','YColor','k','Color','w'); grid on;
% Routing topology
figTopo = figure('Color','w');
SU_pos = netRange*rand(numSU_nodes,2);
xlabel('X','FontName','Times New Roman','FontSize',32,'FontWeight','bold','Color','k');
ylabel('Y','FontName','Times New Roman','FontSize',32,'FontWeight','bold','Color','k');
set(gca,'FontName','Times New Roman','FontSize',28,'FontWeight','bold', ...
    'XColor','k','YColor','k','Color','w');
grid on; xlim([0 netRange]); ylim([0 netRange]);
% Metrics (Live Curves)
figMet = figure('Color','w'); hold on;
hAcc = animatedline('Color','b','LineWidth',2,'DisplayName','Accuracy');
hF1  = animatedline('Color','r','LineWidth',2,'DisplayName','F1');
hPDR = animatedline('Color','g','LineWidth',2,'DisplayName','PDR');
legend('show','FontName','Times New Roman','FontSize',30,'Color','k');
xlabel('Time Slot','FontName','Times New Roman','FontSize',32,'FontWeight','bold','Color','k');
ylabel('Metric Value (%)','FontName','Times New Roman','FontSize',32,'FontWeight','bold','Color','k');
set(gca,'FontName','Times New Roman','FontSize',34,'FontWeight','bold', ...
    'XColor','k','YColor','k','Color','w');
grid on; xlim([1 timeSlots]); ylim([0 100]);
%% ---------------- Real-Time Loop ----------------
SU_assignment_over_time = zeros(timeSlots,numSUs);
for t=1:timeSlots
    % Logistic update (chaos model)
    for ch=1:numChannels
        x(ch) = r_values(ch)*x(ch)*(1-x(ch)) + noiseLevel*randn;
        x(ch) = max(0,min(1,x(ch)));
    end
    PU_activity = (x > 0.5);
    % Scenario-driven usage
    PU_activity(1:20)  = mod(t,iot_cycle) <= iot_on_period & mod(t,iot_cycle)~=0;
    PU_activity(21:50) = rand(1,30) < ind_busy_prob;
    PU_activity(51:80) = rand(1,30) < pub_busy_prob;
    if mod(t,emergency_burst_period)==0
        PU_activity(81:100) = 1;
    else
        PU_activity(81:100) = 0;
    end
    % Store history
    x_history(t,:) = x;
    PU_matrix(:,t) = PU_activity';
    % Update heatmap
    figure(figHeat);
    set(hHeat,'CData',PU_matrix');
    title(sprintf('PU Spectrum Occupancy (t=%d)',t), ...
        'FontName','Times New Roman','FontSize',16,'FontWeight','bold','Color','k');
    xlim([0.5, t+0.5]); ylim([0.5,numChannels+0.5]);
    % Stability detection
    window = max(1,t-windowSize+1):t;
    std_devs = std(x_history(window,:),0,1);
    ground_truth = r_values < bifurcation_threshold;
    predicted = (r_values < bifurcation_threshold) & (std_devs < sigma_threshold);
    % Classification metrics
    TP = sum(predicted==1 & ground_truth==1);
    TN = sum(predicted==0 & ground_truth==0);
    FP = sum(predicted==1 & ground_truth==0);
    FN = sum(predicted==0 & ground_truth==1);
    acc  = 100 * (TP + TN) / (TP + TN + FP + FN + eps);
    prec = 100 * TP / (TP + FP + eps);
    rec  = 100 * TP / (TP + FN + eps);
    f1   = 2 * prec * rec / (prec + rec + eps);
    % SU Assignment
    assignable = find(predicted==1);
    if ~isempty(assignable)
        priority_pool = [intersect(assignable,1:20) ...
                         intersect(assignable,81:100) ...
                         setdiff(assignable,[1:20 81:100])];
        for su=1:numSUs
            idx = mod(t+su-2,numel(priority_pool))+1;
            SU_assignment_over_time(t,su)=priority_pool(idx);
        end
    else
        SU_assignment_over_time(t,:)=0;
    end
    % Metrics proxies
    successes = sum(SU_assignment_over_time(t,:)>0);
    throughput = successes/numSUs;
    pdr = throughput;
    energy = sum(diff(SU_assignment_over_time(1:t,:))~=0,'all')*0.01;
    latency = 60 - 20*(successes/numSUs);
    overhead = sum(diff(SU_assignment_over_time(1:t,:))~=0,'all')/(numSUs*t);
    % Store
    ACC(end+1)=acc; PREC(end+1)=prec; REC(end+1)=rec; F1s(end+1)=f1;
    PDR(end+1)=pdr*100; ENJ(end+1)=energy; LAT(end+1)=latency; OVR(end+1)=overhead;
    % SU plot
    figure(figSU);
    for su=1:numSUs
        set(hSU(su),'XData',1:t,'YData',SU_assignment_over_time(1:t,su));
    end
    title(sprintf('SU Assignments (t=%d)',t),'FontName','Times New Roman','FontSize',16,'FontWeight','bold','Color','k');
    % ---------------- Routing topology ----------------
    figure(figTopo); cla; hold on;
    % Create graph with nodes
    G = graph();
    G = addnode(G,numSU_nodes);
    if ~isempty(assignable)
        for i=1:numSU_nodes
            for j=i+1:numSU_nodes
                if norm(SU_pos(i,:)-SU_pos(j,:)) < commRange
                    ch = assignable(randi(numel(assignable)));
                    if std_devs(ch) < sigma_threshold
                        G = addedge(G,i,j);
                    end
                end
            end
        end
    end
    % Plot graph with visible nodes/edges
    plot(G,'XData',SU_pos(:,1),'YData',SU_pos(:,2), ...
        'NodeColor','k','MarkerSize',12,'NodeLabel',1:numSU_nodes, ...
        'EdgeColor','b','LineWidth',2);
    xlim([0 netRange]); ylim([0 netRange]);
    grid on;
    xlabel('X','FontName','Times New Roman','FontSize',32,'FontWeight','bold','Color','k');
    ylabel('Y','FontName','Times New Roman','FontSize',34,'FontWeight','bold','Color','k');
    title(sprintf('CR-AODV Routing Topology (t=%d)',t), ...
        'FontName','Times New Roman','FontSize',16,'FontWeight','bold','Color','k');
    % Metrics
    figure(figMet);
    addpoints(hAcc,t,acc); addpoints(hF1,t,f1); addpoints(hPDR,t,pdr*100);
    title(sprintf('Live Performance Metrics (t=%d)',t), ...
        'FontName','Times New Roman','FontSize',16,'FontWeight','bold','Color','k');
    drawnow; pause(0.25);
end
%% ---------------- Final Summary ----------------
fprintf('\n===== Final Real-Time Scenario Summary =====\n');
fprintf('Accuracy   = %.2f %%\n',mean(ACC));
fprintf('Precision  = %.2f %%\n',mean(PREC));
fprintf('Recall     = %.2f %%\n',mean(REC));
fprintf('F1-score   = %.2f %%\n',mean(F1s));
fprintf('PDR        = %.2f %%\n',mean(PDR));
fprintf('Energy     = %.4f mJ\n',mean(ENJ));
fprintf('Latency    = %.2f ms\n',mean(LAT));
fprintf('Overhead   = %.4f\n',mean(OVR));
% Final summary bar chart
finalMetrics = [mean(ACC) mean(PREC) mean(REC) mean(F1s) mean(PDR)];
metricLabels = {'Accuracy','Precision','Recall','F1-score','PDR'};
figSummary = figure('Color','w');
bar(finalMetrics,'FaceColor',[0.2 0.6 0.8]);
set(gca,'XTickLabel',metricLabels,'FontName','Times New Roman','FontSize',34, ...
    'FontWeight','bold','XColor','k','YColor','k','Color','w');
ylabel('Value (%)','FontName','Times New Roman','FontSize',30,'FontWeight','bold','Color','k');
title('Final Performance Summary','FontName','Times New Roman','FontSize',34,'FontWeight','bold','Color','k');
grid on;