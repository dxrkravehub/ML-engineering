import streamlit as st
import pandas as pd
import os
import zipfile
import shutil # Import shutil for cleanup

# Импортируем функции из нашего файла расчетов
from calculations import process_zip_file, process_client_data

st.title("Генератор Push-уведомлений для клиентов")

st.write("Загрузите ZIP-архив с данными клиентов (clients.csv) и файлами транзакций/переводов (client_i_transactions_3m.csv, client_i_transfers_3m.csv).")

uploaded_file = st.file_uploader("Загрузить ZIP-файл", type="zip")

if uploaded_file is not None:
    # Сохраняем загруженный файл временно
    zip_file_path = "uploaded_data.zip"
    with open(zip_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Файл успешно загружен. Обработка данных...")

    data_dir, clients_df = process_zip_file(zip_file_path)

    if data_dir and clients_df is not None:
        st.write("Файлы успешно извлечены и clients.csv загружен.")

        # Process client data
        result_df = process_client_data(data_dir, clients_df)

        if not result_df.empty:
            st.write("Результаты обработки:")
            st.dataframe(result_df)

            # Option to download the results
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(result_df)

            st.download_button(
                label="Скачать результаты в CSV",
                data=csv,
                file_name='solution_case1_all_clients.csv',
                mime='text/csv',
            )

        else:
            st.warning("Не удалось обработать данные клиентов. Проверьте формат файлов транзакций/переводов.")

        # Clean up extracted files
        try:
            shutil.rmtree(data_dir)
            os.remove(zip_file_path)
            st.write("Временные файлы очищены.")
        except Exception as e:
            st.warning(f"Не удалось очистить временные файлы: {e}")


    else:
        st.error("Произошла ошибка при извлечении файлов или чтении clients.csv. Убедитесь, что ZIP-архив содержит 'clients.csv'.")
