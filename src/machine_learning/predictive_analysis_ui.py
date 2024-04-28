import streamlit as st


# Function predicts house prices using the regression pipeline
def predict_sale_price(X_live, property_features, sale_price_pipeline):

    # From live data filter most imp. features to ensure
    # correct feature order andto avoid an incorrect prediction.
    X_live_sale_price = X_live.filter(property_features)

    # predict
    sale_price_prediction = sale_price_pipeline.predict(X_live_sale_price)

    statement = (
        f"Estimated sale value:"
        )

    # Format the value dispalyed on the page
    # Formatting code block taken from
    # https://github.com/t-hullis/milestone-project-heritage-housing-issues/tree/main
    if len(sale_price_prediction) == 1:
        price = float(sale_price_prediction.round(1))
        price = '${:,.2f}'.format(price)

        st.write(statement)
        st.write(f"**{price}**")
    else:
        st.write(
            f"* Estimated sale values of inherited real estate:")
        st.write(sale_price_prediction)

    return sale_price_prediction
