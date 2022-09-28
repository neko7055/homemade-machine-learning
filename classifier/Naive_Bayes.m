classdef Naive_Bayes < handle
    
    properties
        p0
        p1 
        TF
        P_one
        P_zero
        w

    end
    
    methods
        function obj = Naive_Bayes()
        end
        function fit(obj,x,y,w)
            y = (y + 1)/2;
            idx1 = (y==1);
            idx0 = (y==0);
            if nargin == 3
                if isempty(obj.w)
                    obj.w = ones(length(y),1)/length(y);
                end
            else
                obj.w = w;
            end
            w0 = obj.w(idx0);
            w1 = obj.w(idx1);
            if isempty(obj.TF)
                obj.TF = minimax_tf(x);
            end
            x = obj.TF.tf(x);
            if isempty(w0)
                obj.p0 = @(x) zeros(size(x,1),1);
                obj.p1 = @(x) ones(size(x,1),1);
                return
            end
            if isempty(w1)
                obj.p0 = @(x) ones(size(x,1),1);
                obj.p1 = @(x) zeros(size(x,1),1);
                return
            end
            obj.P_zero = sum(w0);
            obj.P_one = sum(w1);
            x0 = x(idx0,:);
            x1 = x(idx1,:);
            
            w0_ = w0 ./ sum(w0);
            w1_ = w1 ./ sum(w1);
            mu0 = sum(x0.*w0_,1);
            mu1 = sum(x1.*w1_,1);
            
            var0 = (x0 - mu0)' * diag(w0_)*(x0 - mu0); %sum((x0 - mu0).^2  .*w0_,1);
            var1 = (x1 - mu1)' * diag(w1_)*(x1 - mu1);%sum((x1 - mu1).^2  .*w1_,1);
            obj.p0 = @(x) IDgaussianPDF(x , mu0, (var0));
            obj.p1 = @(x) IDgaussianPDF(x , mu1, (var1));

        end
        function out = copy(obj)
                out = Naive_Bayes();
                out.p0 =obj.p0;
                out.p1  =obj.p1;
                if ~isempty(obj.TF)
                    out.TF =obj.TF.copy();
                end 
                out.P_one =obj.P_one;
                out.P_zero =obj.P_zero;
                out.w = obj.w;
        end
        function y = predict(obj,x)
            x = obj.TF.tf(x);
            pred_p0 = obj.p0(x).*obj.P_zero;
            pred_p1 = obj.p1(x).*obj.P_one;
            %disp(sum((pred_p0 ==0) .*(pred_p1 ==0) ))
            y = double(pred_p1>pred_p0);
            y = 2*y - 1;
        end
    end
end

function out = IDgaussianPDF(x,mu,var)
d = length(mu);
D =det(var);
x_ = x - mu;
%var_inv = inv(var);
out = exp(-sum(x_.*(var\(x_'))',2)./(2))./sqrt((2*pi)^d*abs(D));
end

