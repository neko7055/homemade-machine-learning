classdef KNN < handle
    %KNN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        TF
        x
        y
        k
        w
        max_k 
    end
    
    methods
        function obj = KNN()
        end
        
        function fit(obj,x,y,w,max_k)
            switch nargin 
                case 3
                    if isempty(obj.w)
                        obj.w = ones(size(x,1),1)/size(x,1);
                    end
                    if isempty(obj.max_k)
                        obj.max_k  = 25;
                    end
                case 4
                    obj.w = w;
                    if isempty(obj.max_k)
                        obj.max_k  = 25;
                    end
                otherwise
                    obj.w = w;
                    obj.max_k = max_k;
            end
            if isempty(obj.TF)
                obj.TF = minimax_tf(x);
            end
            obj.x = obj.TF.tf(x);
            obj.y = y;
            Max = -inf;
            Max_k = inf;
            for k = 1:obj.max_k
                obj.k = k;
                pred_y = obj. train_predict(x);
                r = sum(obj.w.*obj.y.*pred_y);
                if r > Max
                    Max = r;
                    Max_k = k;
                end
            end
            obj.k = Max_k;
        end
        function out=copy(obj)
        out = KNN();
        if ~isempty(obj.TF)
            out.TF =obj.TF.copy();
        end
        out.x =obj.x;
        out.y =obj.y;
        out.k =obj.k;
        out.w = obj.w;
        out.max_k = obj.max_k;
        end
        function pred_y = predict(obj,x)
            x = obj.TF.tf(x);
            pred_y = zeros(size(x,1),1);
            for i = 1:size(x,1)
                d_v1 = x(i,:) - obj.x(obj.y==1,:);
                w1 = obj.w(obj.y==1,:);
                d1 = sqrt(sum(d_v1.^2,2));
                [B1 ,I1] = mink(d1,obj.k);
                d_v0 = x(i,:) - obj.x(obj.y==-1,:);
                w0 = obj.w(obj.y==-1,:);
                d0 = sqrt(sum(d_v0.^2,2));
                [B0 ,I0]= mink(d0,obj.k);
                if B1(1) == 0
                    pred_y(i) =1;
                elseif B0(1) ==0
                    pred_y(i) =0;
                else
                    pred_y(i) = double(sum((1./B1).*w1(I1)/sum(w1(I1))) > sum((1./B0).*w0(I0)/sum(w0(I0))));
                end
            end
            pred_y = 2*pred_y - 1;
        end

        function pred_y = train_predict(obj,x)
            x = obj.TF.tf(x);
            pred_y = zeros(size(x,1),1);
            N = size(x,1);
            for i = 1:size(x,1)
                choose_x = obj.x([1:i-1,i+1:N],:);
                choose_y = obj.y([1:i-1,i+1:N]);
                d_v1 = x(i,:) - choose_x(choose_y==1,:);
                w1 = obj.w(choose_y==1,:);
                d1 = sqrt(sum(d_v1.^2,2));
                [B1 ,I1] = mink(d1,obj.k);
                d_v0 = x(i,:) - choose_x(choose_y==-1,:);
                w0 = obj.w(choose_y==-1,:);
                d0 = sqrt(sum(d_v0.^2,2));
                [B0 ,I0]= mink(d0,obj.k);
                if B1(1) == 0
                    pred_y(i) =1;
                elseif B0(1) ==0
                    pred_y(i) =0;
                else
                    pred_y(i) = double(sum((1./B1).*w1(I1)/sum(w1(I1))) > sum((1./B0).*w0(I0)/sum(w0(I0))));
                end
            end
            pred_y = 2*pred_y - 1;
        end
    end
end

