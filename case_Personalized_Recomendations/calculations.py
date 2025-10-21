import pandas as pd
import random
import zipfile
import os

# Настройки категорий
TRAVEL_CATS = {'Путешествия', 'Отели', 'Такси', 'Поезда', 'Самолеты'}
ONLINE_CATS = {'Играем дома', 'Смотрим дома', 'Едим дома', 'Кино', 'Развлечения'}
RESTAURANT_CATS = {'Кафе и рестораны'}
JEWELRY_CATS = {'Ювелирные украшения'}
PERFUME_CATS = {'Косметика и Парфюмерия'}

# Вспомогательные функции
def monthly_sum(group, cats):
    return group.loc[group['category'].isin(cats), 'amount'].sum() / 3.0

def total_monthly_spending(group):
    return group['amount'].sum().sum() / 3.0

def top3_categories_monthly(group):
    s = group.groupby('category')['amount'].sum().sort_values(ascending=False)
    top3 = s.head(3) / 3.0
    return list(top3.index), top3.sum()

def monthly_fx_volume(tr_group):
    fx_types = {'fx_buy', 'fx_sell'}
    return tr_group.loc[tr_group['type'].isin(fx_types), 'amount'].sum() / 3.0

def monthly_cash_card_transfers(tr_group):
    types = {'p2p_out', 'card_out', 'atm_withdrawal'}
    return tr_group.loc[tr_group['type'].isin(types), 'amount'].sum() / 3.0

def relevance_cash_loan(tg, tr_group):
    debt_types = {'loan_payment_out', 'cc_repayment_out', 'installment_payment_out'}
    debts = tr_group.loc[tr_group['type'].isin(debt_types), 'amount'].sum() / 3.0
    total = tg['amount'].sum().sum() / 3.0
    ratio = debts / max(total, 1)
    return ratio >= 0.2

# === Вычисление выгоды по продуктам ===
def calc_benefits(tg, trg, avg_balance):
    benefits = {}
    ctx = {}

    total_spending = total_monthly_spending(tg)

    # Карта для путешествий
    travel_taxi = monthly_sum(tg, TRAVEL_CATS)
    benefits['Карта для путешествий'] = 0.04 * travel_taxi
    travel_percentage = (travel_taxi / max(total_spending, 1)) * 100
    ctx['Карта для путешествий'] = {'travel_taxi': travel_taxi, 'travel_percentage': round(travel_percentage)}


    # Премиальная карта
    restaurants = monthly_sum(tg, RESTAURANT_CATS)
    jewelry = monthly_sum(tg, JEWELRY_CATS)
    perfume = monthly_sum(tg, PERFUME_CATS)
    boosted = restaurants + jewelry + perfume
    other = max(tg['amount'].sum().sum()/3 - boosted, 0)
    if avg_balance >= 6_000_000: rate = 0.04
    elif avg_balance >= 1_000_000: rate = 0.03
    else: rate = 0.02
    cashback = rate * other + 0.04 * boosted
    cashback = min(cashback, 100_000)
    fee_saving = min(0.002 * monthly_cash_card_transfers(trg), 3_000)
    benefits['Премиальная карта'] = cashback + fee_saving
    ctx['Премиальная карта'] = {'cashback': cashback}

    # Кредитная карта
    top3_names, top3_sum = top3_categories_monthly(tg)
    online = monthly_sum(tg, ONLINE_CATS)
    benefits['Кредитная карта'] = 0.07 * top3_sum + 0.10 * online
    ctx['Кредитная карта'] = {'top3': top3_names}

    # Обмен валют
    fx_vol = monthly_fx_volume(trg)
    benefits['Обмен валют'] = 0.003 * fx_vol
    ctx['Обмен валют'] = {'fx_curr': 'USD', 'fx_vol': fx_vol}

    # Депозиты
    if avg_balance > 100000:
        benefits['Депозит Мультивалютный'] = (avg_balance * 0.145) / 12.0
        benefits['Депозит Сберегательный'] = (avg_balance * 0.165) / 12.0
        benefits['Депозит Накопительный'] = (avg_balance * 0.155) / 12.0
    else:
        benefits['Депозит Мультивалютный'] = 0
        benefits['Депозит Сберегательный'] = 0
        benefits['Депозит Накопительный'] = 0

    # Инвестиции
    if avg_balance > 500000:
        benefits['Инвестиции'] = avg_balance * 0.005
    else:
        benefits['Инвестиции'] = 0

    # Кредит наличными
    if relevance_cash_loan(tg, trg):
        benefits['Кредит наличными'] = 1
    else:
        benefits['Кредит наличными'] = 0

    # Золотые слитки
    if avg_balance > 10000000:
         benefits['Золотые слитки'] = avg_balance * 0.001
    else:
         benefits['Золотые слитки'] = 0

    return benefits, ctx

# === Альтернативные шаблоны пушей ===
push_templates = {
    'Карта для путешествий': {
        'young': ["{name}, много поездок/такси в этом месяце? 🚗 С тревел-картой часть расходов вернулась бы кешбэком. Хочешь оформить?",
                  "{name}, вжух-вжух по городу или в поездки? ✈️ Верни часть денег с тревел-картой!"],
        'adult': ["{name}, в этом месяце у вас много поездок/такси. С тревел-картой часть расходов вернулась бы кешбэком. Хотите оформить?"],
        'senior': ["{name}, в этом месяце у вас много поездок/такси. С тревел-картой часть расходов вернулась бы кешбэком. Хотите оформить?"]
    },
    'Премиальная карта': {
         'young': ["{name}, крупный остаток + любишь рестораны? ✨ Премиум-карта даст топ кешбэк и бесплатные снятия!",
                   "{name}, баланс в плюсе и ценишь комфорт? Премиум-карта для тебя. 😉"],
         'adult': ["{name}, у вас стабильно крупный остаток и траты в ресторанах. Премиальная карта даст повышенный кешбэк и бесплатные снятия. Оформить сейчас."],
         'senior': ["{name}, у вас стабильно крупный остаток и траты в ресторанах. Премиальная карта даст повышенный кешбэк и бесплатные снятия. Оформить сейчас."]
    },
    'Кредитная карта': {
        'young': ["{name}, твои топ-категории — {cat1}, {cat2}, {cat3}. 🔥 Кредитка даст до 10% кешбэка и на онлайн-сервисы!",
                  "{name}, залипаешь в {cat1} или {cat2}? Кредитка вернет часть денег + рассрочка есть! 😎"],
        'adult': ["{name}, ваши топ-категории — {cat1}, {cat2}, {cat3}. Кредитная карта даёт до 10% в любимых категориях и на онлайн-сервисы. Оформить карту."],
        'senior': ["{name}, ваши топ-категории — {cat1}, {cat2}, {cat3}. Кредитная карта даёт до 10% в любимых категориях и на онлайн-сервисы. Оформить карту."]
    },
    'Обмен валют': {
        'young': ["{name}, часто меняешь валюту? 💱 В приложении топ-курс и авто-покупка!",
                  "{name}, нужен бакс или евро? В приложении выгодно и без заморочек. 👇"],
        'adult': ["{name}, вы часто меняете валюту. В приложении выгодный курс и авто-покупка при достижении целевого курса. Открыть."],
        'senior': ["{name}, вы часто меняете валюту. В приложении выгодный курс и авто-покупка при достижении целевого курса. Открыть."]
    },
    'Депозит Мультивалютный': {
        'young': ["{name}, думаешь о вкладах в разных валютах? 🤔 Наш Мультивалютный депозит для тебя. Открыть.",
                  "{name}, хочешь хранить деньги в разных валютах? Мультивалютный депозит — изи решение. ✅"],
        'adult': ["{name}, думаете о вкладах в разных валютах? Наш Мультивалютный депозит для вас. Открыть."],
        'senior': ["{name}, думаете о вкладах в разных валютах? Наш Мультивалютный депозит для вас. Открыть."]
    },
    'Депозит Сберегательный': {
        'young': ["{name}, свободные {balance} ₸ могут приносить 16,5% годовых. 💰 Открой вклад!",
                  "{name}, есть свободные {balance} ₸? Пусть работают под 16,5%! 💪"],
        'adult': ["{name}, свободные {balance} ₸ могут приносить 16,5% годовых. Открыть вклад."],
        'senior': ["{name}, свободные {balance} ₸ могут приносить 16,5% годовых. Открыть вклад."]
    },
    'Депозит Накопительный': {
        'young': ["{name}, есть свободные деньги? 📈 Накопительный депозит поможет их приумножить. Давай!",
                  "{name}, хочешь копить под процент? Накопительный депозит — твой вариант. 👇"],
        'adult': ["{name}, у вас есть свободные средства? Депозит Накопительный поможет их приумножить. Открыть депозит."],
        'senior': ["{name}, у вас есть свободные средства? Депозит Накопительный поможет их приумножить. Открыть депозит."]
    },
    'Инвестиции': {
        'young': ["{name}, хочешь, чтобы деньги работали? 🚀 Начни инвестировать без комиссий в первый год!",
                  "{name}, думаешь про инвестиции? Стартуй с нами без лишних трат! 👇"],
        'adult': ["{name}, хотите, чтобы ваши деньги работали? Начните инвестировать с нами без комиссий в первый год. Начать инвестировать."],
        'senior': ["{name}, хотите, чтобы ваши деньги работали? Начните инвестировать с нами без комиссий в первый год. Начать инвестировать."]
    },
    'Кредит наличными': {
        'young': ["{name}, нужна доп сумма? 💸 Оформи кредит наличными быстро и без геморра!",
                  "{name}, бабки нужны срочно? Кредит наличными без залога и справок. Жми! 👉"],
        'adult': ["{name}, нужна дополнительная сумма? Оформите кредит наличными быстро и просто без залога и справок. Оформить кредит."],
        'senior': ["{name}, нужна дополнительная сумма? Оформите кредит наличными быстро и просто без залога и справок. Оформить кредит."]
    },
    'Золотые слитки': {
         'young': ["{name}, шаришь за крипту... а как насчет золота? ✨ Купи слитки 999,9 пробы в нашем банке!",
                   "{name}, хочешь вложиться во что-то надежное? Золотые слитки — классика. 👑"],
         'adult': ["{name}, интересуетесь альтернативными инвестициями? Приобретайте золотые слитки 999,9 пробы в нашем банке. Приобрести золото."],
         'senior': ["{name}, интересуетесь альтернативными инвестициями? Приобретайте золотые слитки 999,9 пробы в нашем банке. Приобрести золото."]
    }
}


#Генерация пуша с вариативностью
def make_push(name, product, ctx, avg_balance, age):
    if age is not None:
        if age <= 30:
            age_group = 'young'
        elif age >= 40:
            age_group = 'senior'
        else:
            age_group = 'adult'
    else:
        age_group = 'adult'

    templates = push_templates.get(product, {}).get(age_group)
    if not templates:
        templates = push_templates.get(product, {}).get('adult')
        if not templates:
             return f"{name}, у нас есть предложение под ваши привычки расходов. Посмотреть детали."

    tpl = random.choice(templates)
    try:
        formatted_balance = f"{avg_balance:,.0f}".replace(",", " ")
        cat1 = ctx.get('top3', ['—','—','—'])[0]
        cat2 = ctx.get('top3', ['—','—','—'])[1]
        cat3 = ctx.get('top3', ['—','—','—'])[2]

        return tpl.format(
            name=name,
            cat1=cat1,
            cat2=cat2,
            cat3=cat3,
            fx_curr=ctx.get('fx_curr', 'валюте'),
            balance=formatted_balance
        )
    except KeyError as e:
        print(f"Error formatting push for product {product} for age group {age_group}: Missing key {e}")
        return f"{name}, у нас есть предложение под ваши привычки расходов. Посмотреть детали."
    except ValueError as e:
         print(f"Error formatting push for product {product} for age group {age_group}: {e}")
         return f"{name}, у нас есть предложение под ваши привычки расходов. Посмотреть детали."

# Function to process the zip file
def process_zip_file(zip_file_path):
    output_dir = "extracted_data"
    os.makedirs(output_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    except zipfile.BadZipFile:
        print("Ошибка: Некорректный ZIP-файл.")
        return None, None
    except FileNotFoundError:
        print(f"Ошибка: ZIP-файл '{zip_file_path}' не найден.")
        return None, None

    clients_file_path = None
    for root, dirs, files in os.walk(output_dir):
        if 'clients.csv' in files:
            clients_file_path = os.path.join(root, 'clients.csv')
            break

    if not clients_file_path:
        print("Ошибка: Файл clients.csv не найден в архиве.")
        return None, None

    try:
        clients_df = pd.read_csv(clients_file_path)
        clients_df.set_index('client_code', inplace=True)
    except Exception as e:
        print(f"Ошибка при чтении clients.csv: {e}")
        return None, None

    return output_dir, clients_df

# Main processing logic
def process_client_data(data_dir, clients_df):
    rows = []
    used_pushes = set()

    for i in range(1, 61):
        tx_file = os.path.join(data_dir, f"client_{i}_transactions_3m.csv")
        tr_file = os.path.join(data_dir, f"client_{i}_transfers_3m.csv")

        # Search for transaction and transfer files recursively
        found_tx = False
        for root, dirs, files in os.walk(data_dir):
            if f"client_{i}_transactions_3m.csv" in files:
                tx_file = os.path.join(root, f"client_{i}_transactions_3m.csv")
                found_tx = True
                break

        found_tr = False
        for root, dirs, files in os.walk(data_dir):
             if f"client_{i}_transfers_3m.csv" in files:
                 tr_file = os.path.join(root, f"client_{i}_transfers_3m.csv")
                 found_tr = True
                 break


        if not found_tx or not found_tr:
            print(f"Файлы транзакций или переводов не найдены для клиента {i}. Пропускаем.")
            continue

        try:
            tg = pd.read_csv(tx_file)
            trg = pd.read_csv(tr_file)
        except Exception as e:
            print(f"Ошибка при чтении файлов для клиента {i}: {e}. Пропускаем.")
            continue

        name = tg['name'].iloc[0]
        avg_balance = tg['avg_monthly_balance_KZT'].iloc[0] if 'avg_monthly_balance_KZT' in tg.columns else 0
        client_status = clients_df.loc[i, 'status'] if i in clients_df.index and 'status' in clients_df.columns else None
        client_age = clients_df.loc[i, 'age'] if i in clients_df.index and 'age' in clients_df.columns else None

        benefits, ctx = calc_benefits(tg, trg, avg_balance)
        filtered = {k: v for k, v in benefits.items() if v > 0}

        best_product = None
        if filtered:
            sorted_benefits = sorted(filtered.items(), key=lambda item: item[1], reverse=True)
            top_n_products = [item[0] for item in sorted_benefits[:4]]

            product_weights = {}
            for rank, product in enumerate(top_n_products):
                weight = (4 - rank)
                if client_status == 'Премиальный клиент' and product in ['Премиальная карта', 'Золотые слитки', 'Инвестиции']:
                     weight += 2
                elif client_status == 'Студент' and product in ['Кредитная карта']:
                     weight += 2
                elif client_status == 'Зарплатный клиент' and product in ['Кредитная карта', 'Карта для путешествий']:
                     weight += 1

                product_weights[product] = weight

            available_products = [p for p in product_weights.keys() if p in push_templates and (push_templates[p].get('young') or push_templates[p].get('adult') or push_templates[p].get('senior'))]

            if available_products:
                best_product = random.choices(
                    available_products,
                    weights=[product_weights[p] for p in available_products],
                    k=1
                )[0]
            elif top_n_products:
                 best_product = sorted_benefits[0][0]
            else:
                 best_product = max(benefits, key=benefits.get)

        else:
            best_product = max(benefits, key=benefits.get)

        push = make_push(name, best_product, ctx.get(best_product, {}), avg_balance, client_age)

        attempts = 0
        initial_push = push
        while push in used_pushes and attempts < 3:
            push = make_push(name, best_product, ctx.get(best_product, {}), avg_balance, client_age)
            if push == initial_push:
                 break
            attempts += 1

        used_pushes.add(push)
        rows.append({'client_code': i, 'product': best_product, 'push_notification': push})

    return pd.DataFrame(rows)
