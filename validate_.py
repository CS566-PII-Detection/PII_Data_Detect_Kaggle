def validate(model, valid_x, valid_y):
    tp_data=[]
    tn_data=[]
    fp_data=[]
    fn_data=[]
    pred_y = model.predict(valid_x)
    for i in range(len(pred_y)):
        if pred_y[i]==valid_y[i]==1:
            tp_data.append(valid_x[i])
        elif pred_y[i]==valid_y[i]==0:
            tn_data.append(valid_x[i])
        elif pred_y[i]==0 and valid_y[i]==1:
            fn_data.append(valid_x[i])
        elif pred_y[i]==1 and valid_y[i]==0:
            fp_data.append(valid_x[i])
    return tp_data, tn_data, fp_data, fn_data
