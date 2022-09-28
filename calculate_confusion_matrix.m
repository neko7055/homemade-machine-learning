function out = calculate_confusion_matrix(y,pred_y)
y_idx_Y = y==1;
y_idx_N = y==-1;
pred_y_idx_Y = pred_y == 1;
pred_y_idx_N = pred_y == -1;
TN = sum((pred_y_idx_N) & (y_idx_N));
TP = sum((pred_y_idx_Y) & (y_idx_Y));
FN =sum((pred_y_idx_N) & (y_idx_Y)) ;
FP = sum((pred_y_idx_Y) & (y_idx_N)) ;

confusion_matrix = [TP,FP;
                                    FN,TN];

out = confusion_matrix;
end