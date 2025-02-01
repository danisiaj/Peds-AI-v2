import streamlit as st
    
def save_changes(temperature, max_tokens):
    with st.spinner('Saving...'):
        st.session_state.temperature = temperature
        st.session_state.max_tokens = max_tokens
    st.success('Changes saved!')
    st.write(st.session_state.temperature, st.session_state.max_tokens)

def set_up_page():
    st.header('Settings')
    st.image('images/logo_3_copy.png', width=100)
    st.markdown(f'##### _Role:_         {st.session_state.role}')
    col1, col2 = st.columns([1,1])
    with col1:
        temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    with col2:
        max_tokens = st.slider('Max tokens', min_value=0, max_value=10000, value=2000)
    st.button('Save changes', on_click=save_changes, args=(temperature, max_tokens))
    




def main():
    set_up_page()


main()