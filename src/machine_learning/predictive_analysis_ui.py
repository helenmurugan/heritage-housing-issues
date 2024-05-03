import streamlit as st


# Function predicts house prices using the regression pipeline
def predict_sale_price(X_live, property_features, sale_price_pipeline):

    # From live data filter most imp. features to ensure
    # correct feature order and avoid an incorrect prediction.
    X_live_sale_price = X_live.filter(property_features)

    # predict
    sale_price_prediction = sale_price_pipeline.predict(X_live_sale_price)

    # Format the value displayed on the page
    if len(sale_price_prediction) == 1:
        price = float(sale_price_prediction.round(0))
        price = '${:,.0f}'.format(price)
        st.write(f"Sale price prediction:")
        st.write(f"**{price}**")
    else:
        st.write(
            f"Sales price predictions for inherited properties:")
        st.write(sale_price_prediction)

    return sale_price_prediction
