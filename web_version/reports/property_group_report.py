import pandas as pd
# Removed psycopg2 imports
from sqlalchemy import text # Added for SQL query execution with parameters if needed
from ..utils import get_engine
import streamlit as st # Import streamlit at the top

def generate_property_group_data(start_date=None, end_date=None):
    """
    Fetches data for active properties and active units, separating grouped and ungrouped.
    Returns a dictionary containing four DataFrames:
    'grouped_properties', 'ungrouped_properties', 'grouped_units', 'ungrouped_units'.
    """
    engine = get_engine()
    if engine is None:
        st.error("Database connection failed.")
        return None

    results = {
        'grouped_properties': pd.DataFrame(),
        'ungrouped_properties': pd.DataFrame(),
        'grouped_units': pd.DataFrame(),
        'ungrouped_units': pd.DataFrame()
    }
    try:
        # --- Property Queries ---
        # Query 1: Get properties WITH group assignments
        sql_grouped_props = text("""
            SELECT DISTINCT -- Ensure one row per property per group it's in
                pg.name AS "Property Group",
                rp.id AS "Property ID",
                rp.name AS "Property Name"
            FROM
                rental_properties rp
            JOIN
                property_group_memberships pgm ON rp.id = pgm.property_id
            JOIN
                property_groups pg ON pgm.property_group_id = pg.id
            WHERE
                rp.is_active = true
            ORDER BY
                "Property Group",
                "Property Name";
        """)
        results['grouped_properties'] = pd.read_sql_query(sql_grouped_props, engine)

        # Query 2: Get properties WITHOUT group assignments
        sql_ungrouped_props = text("""
            SELECT
                rp.id AS "Property ID",
                rp.name AS "Property Name"
            FROM
                rental_properties rp
            LEFT JOIN
                property_group_memberships pgm ON rp.id = pgm.property_id
            WHERE
                rp.is_active = true
                AND pgm.property_id IS NULL -- Key condition for ungrouped
            ORDER BY
                "Property Name";
        """)
        results['ungrouped_properties'] = pd.read_sql_query(sql_ungrouped_props, engine)

        # --- Unit Queries ---
        # Query 3: Get units belonging to properties WITH group assignments
        sql_grouped_units = text("""
            SELECT
                pg.name AS "Property Group",
                rp.id AS "Property ID",
                rp.name AS "Property Name",
                ru.id AS "Unit ID",
                ru.unit_number AS "Unit Number"
            FROM
                rental_units ru
            JOIN
                rental_properties rp ON ru.property_id = rp.id
            JOIN
                property_group_memberships pgm ON rp.id = pgm.property_id
            JOIN
                property_groups pg ON pgm.property_group_id = pg.id
            WHERE
                rp.is_active = true
                AND ru.is_active = true -- Only active units
            ORDER BY
                "Property Group",
                "Property Name",
                CASE -- Attempt numeric sort for unit numbers
                    WHEN ru.unit_number ~ E'^\\\\d+$' THEN LPAD(ru.unit_number, 10, '0')
                    ELSE ru.unit_number
                END;
        """)
        results['grouped_units'] = pd.read_sql_query(sql_grouped_units, engine)

        # Query 4: Get units belonging to properties WITHOUT group assignments
        sql_ungrouped_units = text("""
            SELECT
                rp.id AS "Property ID",
                rp.name AS "Property Name",
                ru.id AS "Unit ID",
                ru.unit_number AS "Unit Number"
            FROM
                rental_units ru
            JOIN
                rental_properties rp ON ru.property_id = rp.id
            LEFT JOIN
                property_group_memberships pgm ON rp.id = pgm.property_id
            WHERE
                rp.is_active = true
                AND ru.is_active = true -- Only active units
                AND pgm.property_id IS NULL -- Properties not in any group
            ORDER BY
                "Property Name",
                CASE -- Attempt numeric sort for unit numbers
                    WHEN ru.unit_number ~ E'^\\\\d+$' THEN LPAD(ru.unit_number, 10, '0')
                    ELSE ru.unit_number
                END;
        """)
        results['ungrouped_units'] = pd.read_sql_query(sql_ungrouped_units, engine)


        return results

    except Exception as error:
        st.error(f"Error fetching property group data: {error}")
        import logging # Consider adding logging here
        logging.exception("Error in generate_property_group_data")
        return None

def main(start_date, end_date):
    """
    Main function to generate and display the property group report in Streamlit.
    Shows counts per group (properties and units), bar charts, and lists ungrouped properties/units.
    """
    st.title("Property & Unit Group Membership Report")
    st.write("Overview of active properties and units by property group assignment.")

    # Generate the report data
    report_data = generate_property_group_data(start_date, end_date)

    if report_data is None:
        return # Error message already shown

    df_grouped_props = report_data['grouped_properties']
    df_ungrouped_props = report_data['ungrouped_properties']
    df_grouped_units = report_data['grouped_units']
    df_ungrouped_units = report_data['ungrouped_units']

    # --- Filter out the specified group ---
    group_to_exclude = "Elena Morris Management Group 1"
    df_grouped_props_filtered = df_grouped_props[df_grouped_props['Property Group'] != group_to_exclude].copy()
    df_grouped_units_filtered = df_grouped_units[df_grouped_units['Property Group'] != group_to_exclude].copy()

    if df_grouped_props_filtered.empty and df_ungrouped_props.empty:
        st.warning("No active properties found (after filtering).")
        # Optionally check units too, but property check is likely sufficient
        # return

    # --- Property Analysis ---
    st.header("Property Analysis by Group")
    st.subheader("Property Counts per Group")
    if not df_grouped_props_filtered.empty:
        prop_group_counts = df_grouped_props_filtered['Property Group'].value_counts().sort_index()
        prop_group_counts_df = prop_group_counts.reset_index()
        prop_group_counts_df.columns = ['Property Group', 'Property Count']
        st.dataframe(prop_group_counts_df, use_container_width=True, hide_index=True)

        st.subheader("Property Distribution by Group")
        st.bar_chart(prop_group_counts)

        with st.expander("View Details of Grouped Properties (Filtered)"):
            st.dataframe(df_grouped_props_filtered, use_container_width=True, hide_index=True)
            csv_grouped_props = df_grouped_props_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered Grouped Properties as CSV",
                data=csv_grouped_props,
                file_name='grouped_properties_filtered_report.csv',
                mime='text/csv',
                key='grouped_props_csv'
            )
    else:
        st.info("No properties are assigned to the displayed property groups.")

    # --- Unit Analysis ---
    st.header("Unit Analysis by Group")
    st.subheader("Unit Counts per Group")
    if not df_grouped_units_filtered.empty:
        unit_group_counts = df_grouped_units_filtered['Property Group'].value_counts().sort_index()
        unit_group_counts_df = unit_group_counts.reset_index()
        unit_group_counts_df.columns = ['Property Group', 'Unit Count']
        st.dataframe(unit_group_counts_df, use_container_width=True, hide_index=True)

        st.subheader("Unit Distribution by Group")
        st.bar_chart(unit_group_counts)

        with st.expander("View Details of Grouped Units (Filtered)"):
            st.dataframe(df_grouped_units_filtered, use_container_width=True, hide_index=True)
            csv_grouped_units = df_grouped_units_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered Grouped Units as CSV",
                data=csv_grouped_units,
                file_name='grouped_units_filtered_report.csv',
                mime='text/csv',
                key='grouped_units_csv'
            )
    else:
        st.info("No active units found in the displayed property groups.")


    # --- Ungrouped Section ---
    st.header("Ungrouped Items")

    # Display Ungrouped Properties
    st.subheader("Properties Not Assigned to Any Group")
    if not df_ungrouped_props.empty:
        st.dataframe(df_ungrouped_props, use_container_width=True, hide_index=True)
        csv_ungrouped_props = df_ungrouped_props.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Ungrouped Properties as CSV",
            data=csv_ungrouped_props,
            file_name='ungrouped_properties_report.csv',
            mime='text/csv',
            key='ungrouped_props_csv'
        )
    else:
        st.info("All active properties are assigned to at least one group.")

    # Display Ungrouped Units (Units whose properties are not in a group)
    st.subheader("Units in Ungrouped Properties")
    if not df_ungrouped_units.empty:
        st.dataframe(df_ungrouped_units, use_container_width=True, hide_index=True)
        csv_ungrouped_units = df_ungrouped_units.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Units in Ungrouped Properties as CSV",
            data=csv_ungrouped_units,
            file_name='ungrouped_units_report.csv',
            mime='text/csv',
            key='ungrouped_units_csv'
        )
    else:
        st.info("No active units found in ungrouped properties.")


if __name__ == '__main__':
    # This section is primarily for non-Streamlit testing if needed
    print("Testing data generation function (run with Streamlit for full report):")
    data = generate_property_group_data()
    if data:
        print("\nGrouped Properties Sample:")
        print(data['grouped_properties'].head().to_string())
        print("\nUngrouped Properties Sample:")
        print(data['ungrouped_properties'].head().to_string())
        print("\nGrouped Units Sample:")
        print(data['grouped_units'].head().to_string())
        print("\nUngrouped Units Sample:")
        print(data['ungrouped_units'].head().to_string())

    else:
        print("Error generating data.")
