import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import json
import warnings

warnings.filterwarnings("ignore")


class Session2Tasks:
    """Класс для генерации артефактов 2 сессии (Диаграммы, ERD, UI, API) средствами Python."""

    @staticmethod
    def draw_box(ax, text, x, y, width, height, boxstyle="round,pad=0.3", fc="white", fontsize=10, ha="center",
                 va="center"):
        """Вспомогательный метод для отрисовки блоков с текстом."""
        ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
                bbox=dict(boxstyle=boxstyle, fc=fc, ec="black", lw=1.5), zorder=3)

    @staticmethod
    def draw_line(ax, x1, y1, x2, y2, text="", text_pos=0.5):
        """Вспомогательный метод для отрисовки соединительных линий с подписями."""
        ax.plot([x1, x2], [y1, y2], color="black", lw=1.5, zorder=1)
        if text:
            mx, my = x1 + (x2 - x1) * text_pos, y1 + (y2 - y1) * text_pos
            ax.text(mx, my + 0.02, text, ha="center", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

    def generate_use_case_diagram(self, filename="Session2_UseCaseDiagram.pdf"):
        """2.1 Диаграмма вариантов использования (Use Case)."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Use Case Diagram: Belle Croissant Lyonnais", fontsize=16, fontweight='bold', pad=20)

        # Границы системы
        rect = patches.Rectangle((0.3, 0.1), 0.6, 0.8, linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(0.6, 0.87, "System", ha="center", fontsize=12, fontweight='bold', color='grey')

        # Акторы
        self.draw_box(ax, "Staff\n(Actor)", 0.15, 0.7, 0.1, 0.1, boxstyle="circle", fc="lightblue")
        self.draw_box(ax, "Customer\n(Actor)", 0.15, 0.3, 0.1, 0.1, boxstyle="circle", fc="lightgreen")

        # Варианты использования
        use_cases = [
            (0.6, 0.75, "View Dashboard\n& Analytics"),
            (0.6, 0.60, "Manage Orders\n(In-store & Online)"),
            (0.6, 0.45, "Manage Inventory\n& Ingredients"),
            (0.6, 0.30, "Manage Customers\n& Loyalty"),
            (0.6, 0.15, "Place Online Order")
        ]

        for x, y, text in use_cases:
            self.draw_box(ax, text, x, y, 0.2, 0.08, boxstyle="ellipse,pad=0.5", fc="#f9f9f9")

        # Связи Staff
        self.draw_line(ax, 0.2, 0.7, 0.45, 0.75)
        self.draw_line(ax, 0.2, 0.7, 0.45, 0.60)
        self.draw_line(ax, 0.2, 0.7, 0.45, 0.45)
        self.draw_line(ax, 0.2, 0.7, 0.45, 0.30)

        # Связи Customer
        self.draw_line(ax, 0.2, 0.3, 0.45, 0.15)

        # Customer участвует в программе лояльности (косвенно связан с профилем)
        ax.plot([0.2, 0.45], [0.3, 0.30], color="black", lw=1.5, linestyle=":")

        with PdfPages(filename) as pdf:
            pdf.savefig(fig)
        plt.close()
        print(f"[{filename}] успешно сгенерирован.")

    def generate_erd(self, filename="Session2_ERD.pdf"):
        """2.2 Entity-Relationship Diagram (ERD)."""
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Entity-Relationship Diagram (ERD) - 3NF", fontsize=16, fontweight='bold', pad=20)

        # Сущности (Таблицы)
        tables = {
            "CUSTOMER": (0.2, 0.7,
                         "PK  customer_id (INT)\n---\nname (VARCHAR)\nphone (VARCHAR)\nloyalty_points (INT)\njoin_date (DATE)"),
            "ORDER": (0.8, 0.7,
                      "PK  order_id (INT)\nFK  customer_id (INT)\n---\norder_date (DATETIME)\ntotal_amount (DECIMAL)\nstatus (VARCHAR)"),
            "ORDER_ITEM": (0.8, 0.3,
                           "PK  order_item_id (INT)\nFK  order_id (INT)\nFK  product_id (INT)\n---\nquantity (INT)\nunit_price (DECIMAL)"),
            "PRODUCT": (0.2, 0.3,
                        "PK  product_id (INT)\n---\nname (VARCHAR)\ncategory (VARCHAR)\ncost (DECIMAL)\nprice (DECIMAL)\nstock_level (INT)")
        }

        for title, (x, y, attrs) in tables.items():
            content = f"{title}\n{'-' * 25}\n{attrs}"
            self.draw_box(ax, content, x, y, 0.2, 0.2, boxstyle="square,pad=0.5", fc="#fffacc", ha="left", va="center")

        # Связи (Relationship Lines)
        # Customer (1) to Order (N)
        self.draw_line(ax, 0.35, 0.7, 0.65, 0.7, text="1 : N")

        # Order (1) to Order_Item (N)
        self.draw_line(ax, 0.8, 0.55, 0.8, 0.45, text="1 : N")

        # Product (1) to Order_Item (N)
        self.draw_line(ax, 0.35, 0.3, 0.65, 0.3, text="1 : N")

        with PdfPages(filename) as pdf:
            pdf.savefig(fig)
        plt.close()
        print(f"[{filename}] успешно сгенерирован.")

    def generate_wireframes(self, filename="Session2_Wireframes_Staff.pdf"):
        """2.3 UI Wireframes (Макеты интерфейса персонала)."""
        with PdfPages(filename) as pdf:
            # --- Экран 1: Dashboard ---
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title("Wireframe: Staff Dashboard", fontsize=14, fontweight='bold')
            # Sidebar
            ax.add_patch(patches.Rectangle((0, 0), 0.2, 1, fc='#34495e', ec='black'))
            self.draw_box(ax, "Belle Croissant\nAdmin", 0.1, 0.9, 0, 0, fc="none", boxstyle="square,pad=0", ha="center",
                          fontsize=12)
            self.draw_box(ax, "> Dashboard\n  Orders\n  Inventory\n  Customers", 0.1, 0.6, 0, 0, fc="none",
                          boxstyle="square,pad=0", ha="center")
            # Header
            ax.add_patch(patches.Rectangle((0.2, 0.9), 0.8, 0.1, fc='#ecf0f1', ec='black'))
            ax.text(0.22, 0.95, "Dashboard Overview", va="center", fontsize=14)
            # Content Widgets
            self.draw_box(ax, "Total Sales Today\n$1,240.00", 0.35, 0.75, 0, 0, boxstyle="square,pad=1", fc="#e8f8f5")
            self.draw_box(ax, "Active Orders\n12", 0.60, 0.75, 0, 0, boxstyle="square,pad=1", fc="#fef9e7")
            self.draw_box(ax, "Low Stock Alerts\n3 Items", 0.85, 0.75, 0, 0, boxstyle="square,pad=1", fc="#fdedec")
            # Chart Area
            ax.add_patch(patches.Rectangle((0.25, 0.1), 0.7, 0.5, fc='#f2f4f4', ec='grey', linestyle='--'))
            ax.text(0.6, 0.35, "[ Sales Trend Chart Area ]", ha="center", color="grey")
            pdf.savefig(fig)
            plt.close()

            # --- Экран 2: Order Management ---
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title("Wireframe: Order Management", fontsize=14, fontweight='bold')
            # Sidebar
            ax.add_patch(patches.Rectangle((0, 0), 0.2, 1, fc='#34495e', ec='black'))
            self.draw_box(ax, "  Dashboard\n> Orders\n  Inventory\n  Customers", 0.1, 0.6, 0, 0, fc="none",
                          boxstyle="square,pad=0", ha="center")
            # Header
            ax.add_patch(patches.Rectangle((0.2, 0.9), 0.8, 0.1, fc='#ecf0f1', ec='black'))
            ax.text(0.22, 0.95, "Manage Orders", va="center", fontsize=14)
            self.draw_box(ax, "+ Create New Order", 0.85, 0.95, 0, 0, boxstyle="round,pad=0.3", fc="#aed6f1")
            # Table
            ax.add_patch(patches.Rectangle((0.25, 0.1), 0.7, 0.75, fc='white', ec='black'))
            ax.plot([0.25, 0.95], [0.75, 0.75], color="black")  # Table Header Line
            ax.text(0.27, 0.8, "Order ID | Customer | Date | Total | Status | Action", va="center", fontsize=10,
                    fontweight="bold")
            ax.text(0.27, 0.65, "#1001    | John D.  | Today| $15  | Prep   | [Edit]", va="center", fontsize=10)
            ax.text(0.27, 0.55, "#1002    | Sarah K. | Today| $24  | Ready  | [Complete]", va="center", fontsize=10)
            pdf.savefig(fig)
            plt.close()

        print(f"[{filename}] успешно сгенерирован (содержит несколько экранов).")

    def generate_api_design(self, filename="Session2_CustomerAPI_Design.txt"):
        """2.4 REST API Разработка."""
        api_content = """=== Belle Croissant Lyonnais: Customer Management REST API ===

Base URL: /api/v1/customers
Description: Управление данными клиентов и программой лояльности.

---------------------------------------------------------
1. GET /api/v1/customers
Описание: Получить список всех клиентов.
Параметры URL (опционально):
  - page (int): номер страницы (по умолчанию 1)
  - limit (int): количество записей (по умолчанию 20)
  - search (string): поиск по имени или телефону

Response (200 OK):
{
  "status": "success",
  "data": [
    {
      "customer_id": 1,
      "name": "Alex Johnson",
      "phone": "+1234567890",
      "loyalty_points": 150
    }
  ]
}

---------------------------------------------------------
2. GET /api/v1/customers/{id}
Описание: Получить информацию о конкретном клиенте.
Параметры пути:
  - id (int, required): уникальный идентификатор клиента

Response (200 OK):
{
  "status": "success",
  "data": {
    "customer_id": 1,
    "name": "Alex Johnson",
    "phone": "+1234567890",
    "loyalty_points": 150,
    "join_date": "2023-01-15"
  }
}
Response (404 Not Found): {"error": "Customer not found"}

---------------------------------------------------------
3. POST /api/v1/customers
Описание: Создать нового клиента (регистрация).
Body (application/json):
{
  "name": "string, required, max 100 chars",
  "phone": "string, required, regex: ^\\+?[0-9]{10,15}$"
}

Response (201 Created):
{
  "status": "created",
  "customer_id": 2,
  "message": "Customer registered successfully"
}

---------------------------------------------------------
4. PUT /api/v1/customers/{id}
Описание: Обновить данные существующего клиента.
Параметры пути:
  - id (int, required)
Body (application/json):
{
  "name": "string, optional",
  "phone": "string, optional",
  "loyalty_points": "int, optional"
}

Response (200 OK): {"status": "success", "message": "Customer updated"}

---------------------------------------------------------
5. DELETE /api/v1/customers/{id}
Описание: Удалить профиль клиента.
Параметры пути:
  - id (int, required)

Response (204 No Content)
"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(api_content)
        print(f"[{filename}] успешно сгенерирован.")


if __name__ == "__main__":
    generator = Session2Tasks()
    print("Начинаем генерацию артефактов для Сессии 2...")

    # 2.1 Use Case
    generator.generate_use_case_diagram()

    # 2.2 ERD
    generator.generate_erd()

    # 2.3 Wireframes (UI Design)
    generator.generate_wireframes()

    # 2.4 API Design
    generator.generate_api_design()

    print("Все задачи Сессии 2 успешно выполнены!")