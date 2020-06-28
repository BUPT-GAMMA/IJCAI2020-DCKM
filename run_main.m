clearvars;
inPara.numCluster = 10;
inPara.r = 0.5;
inPara.thresh = 0.01;
dataset_set = {'caltech', 'amazon', 'webcam', 'dslr'}
%dataset_set = {'OH1', 'OH2', 'OH3'}
for tt = 1:length(dataset_set)
    dataset = dataset_set{tt};

    load(strcat('./case_study/',dataset,'.mat'));
    %run('./cluster_data/data_all.m');
    %pX(pX>0) = 1;
    [outX_train] = data_filter(X_train);
    size(outX_train)
    inXCell = {outX_train'};
    N1 = size(outX_train, 1);

    Y_all_train = label;
    %Y_all_train = [ones(1000,1);2*ones(660, 1);3*ones(814,1);4*ones(940,1);5*ones(462,1);6*ones(940,1);7*ones(1090,1);8*ones(856,1);9*ones(798,1);10*ones(860,1)];

    %Y_all_train = [ones(940, 1);2*ones(940,1)];
    gt_train = double(Y_all_train);

    lambda1_set = [0.01, 0.1, 1, 10, 100, 1000]
    lambda2_set = [0.01, 0.1, 1, 10, 100, 1000]

    for ii = 1:length(lambda1_set)
        for jj = 1:length(lambda2_set)
            times=1;
            inPara.maxIter = 20;
            lambda0 = 1;
            lambda1 = lambda1_set(ii);
            lambda2 = lambda2_set(jj);
            lambda3 = 1;
            lambda4 = 1;
            nmi_scores_train = zeros(1, times);
            ari_scores_train = zeros(1, times);
            for rr = 1:times
                fprintf('processing iteration %d...\n', rr);

                inG0 = inG(N1, inPara.numCluster);
                [ outG0, outW, outFCell, outAlpha, outAlpha_r, outObj, outNumIter ] = causally_weighted_robust_multi_kmeans( inXCell, inPara, inG0, lambda0, lambda1, lambda2, lambda3, lambda4);
                    
                num = size(outG0, 1);
                Y = outG0*[1 2 3 4 5 6 7 8 9 10]';
                ACCi(rr) = AccMeasure(Y,gt_train);
                [Fi(rr),Pi(rr),Ri(rr)] = compute_f(gt_train, Y);
                nmi_scores_train(rr) = nmi(Y, gt_train);
                ari_scores_train(rr) = adjrand(Y, gt_train);

                fprintf(' nmi_score_train %.4f, ari_score_train %.4f \n', nmi_scores_train(rr), ari_scores_train(rr));
            end
            nmi_mean_train = mean(nmi_scores_train);
            nmi_std_train = std2(nmi_scores_train);
            ari_mean_train = mean(ari_scores_train);
            ari_std_train = std2(ari_scores_train);
            acc_mean_train = mean(ACCi);
            acc_std_train = std2(ACCi);
            Fi_mean_train = mean(Fi);
            Fi_std_train = std2(Fi);
            Pi_mean_train = mean(Pi);
            Pi_std_train = std2(Pi);


            fprintf('nmi_mean_train %.4f, nmi_std_train %.4f \n', nmi_mean_train,nmi_std_train);
            fprintf('ari_mean_train %.4f, ari_std_train %.4f \n', ari_mean_train, ari_std_train);
            fprintf('acc_mean_train %.4f, acc_std_train %.4f \n', acc_mean_train,acc_std_train);
            fprintf('Fi_mean_train %.4f, Fi_std_train %.4f \n', Fi_mean_train, Fi_std_train);
            fprintf('Pi_mean_train %.4f, Pi_std_train %.4f \n', Pi_mean_train,Pi_std_train);

            fprintf('lambda1 %.4f', lambda1)
            fprintf('lambda2 %.4f', lambda2)
            fprintf('lambda3 %.4f', lambda3)
            fprintf('lambda4 %.4f', lambda4)
            save(strcat('./para_ans/',dataset,'_',num2str(lambda1),'_',num2str(lambda2),'.txt'),'nmi_mean_train','nmi_std_train','ari_mean_train','ari_std_train','acc_mean_train','acc_std_train','Fi_mean_train','Fi_std_train','Pi_mean_train','Pi_std_train','-ascii')
        end
    end
end

%fprintf('nmi_mean_test %.4f, nmi_std_test %.4f \n', nmi_mean_test,nmi_std_test);
%fprintf('ari_mean_test %.4f, ari_std_test %.4f \n', ari_mean_test, ari_std_test);
%fprintf('acc_mean_test %.4f, acc_std_test %.4f \n', acc_mean_train,acc_std_train);
%fprintf('Fi_mean_test %.4f, Fi_std_test %.4f \n', Fi_mean_train, Fi_std_train);
%fprintf('Pi_mean_test %.4f, Pi_std_test %.4f \n', Pi_mean_train,Pi_std_train);
%fprintf('lambda1:%.4f', lambda1)
%save('./outW_all.txt', 'outW', '-ascii');
%save('./X_all_train.txt', 'outX_train', '-ascii');
%save('./data/X_all_test.txt', 'X_all_test', '-ascii');
%save('./Y_all_train.txt', 'Y_all_train', '-ascii');
%save('./data/Y_all_test.txt', 'Y_all_test', '-ascii');
