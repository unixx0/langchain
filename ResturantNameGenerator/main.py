import streamlit as st
import working_langchaincode

st.title("Restaurant Name Generator")
cuisine= st.sidebar.selectbox("Pick a Cuisine",("Indian", "Italian", "Mexican", "Arabic", "American", "Chinese", "Korean"))


if cuisine:
    response= working_langchaincode.generate_restaurant_name(cuisine)

    st.header(response["restaurant_name"].strip())
    menu_items= response["menu_items"].strip().split(",")

    st.write("**Menu Items**")
    for items in menu_items:
        st.write("-",items)

