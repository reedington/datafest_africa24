*Database Overview*
The data warehouse is designed to store and manage educational data that provides insights into student performance, behavioral patterns, and factors affecting academic success. The warehouse will facilitate analytical queries and reporting to inform educational stakeholders on how to improve student outcomes.

*Schema Design*
The database schema consists of six primary tables, each representing distinct aspects of the student data. The relationships among these tables are structured to ensure data integrity and facilitate efficient querying.

*Tables and Columns*

1. Student Information Table

Table Name: student_information
Description: Contains basic demographic and academic information about students.
Columns:
StudentID (INT, Primary Key): Unique identifier for each student.
Age (INT): Age of the student.
Gender (VARCHAR): Gender of the student.
StateOfOrigin (VARCHAR): State where the student is from.
EconomicBackground (VARCHAR): Economic status (e.g., low, medium, high).
AttendanceRate (FLOAT): Percentage of classes attended.
Class (VARCHAR): Class or grade level of the student.
OverallPercentage (FLOAT): Cumulative percentage score across subjects.
2. Behavioral and Socioeconomic Factors Table

Table Name: behavioral_and_socioeconomic_factors
Description: Stores behavioral and socioeconomic metrics for students.
Columns:
StudentID (INT, Foreign Key): References student_information.StudentID.
StudyTimeWeekly (INT): Weekly hours spent studying.
ParentalSupportLevel (INT): Level of support from parents (scale of 1-5).
BehavioralScore (FLOAT): Score representing behavioral assessment.
TeacherStudentRatio (FLOAT): Ratio of teachers to students.
Tutoring (BOOLEAN): Indicates whether the student receives tutoring.
Absences (INT): Number of absences in the academic year.
ParentalEducation (VARCHAR): Education level of the parents.
3. Academic Performance Table

Table Name: academic_performance
Description: Records academic performance across various subjects and terms.
Columns:
StudentID (INT, Foreign Key): References student_information.StudentID.
AcademicYear (YEAR): The academic year of the performance record.
Term (VARCHAR): The term of the academic year (e.g., First, Second, Third).
Subject (VARCHAR): Subject name (e.g., Mathematics, English).
Score (FLOAT): Score obtained in the subject.
Class (VARCHAR): Class or grade level of the student during that term.
4. Performance Prediction Labels Table

Table Name: performance_prediction_labels
Description: Contains labels for predicting student performance outcomes.
Columns:
StudentID (INT, Foreign Key): References student_information.StudentID.
AcademicYear (YEAR): The academic year for the prediction label.
RiskOfFailure (BOOLEAN): Indicates risk of failure (True/False).
ImprovementFlag (BOOLEAN): Indicates if improvement is noted (True/False).
5. Extracurricular Activities Table

Table Name: extracurricular_activities
Description: Records students’ participation in extracurricular activities.
Columns:
StudentID (INT, Foreign Key): References student_information.StudentID.
ActivityType (VARCHAR): Type of extracurricular activity (e.g., Sports, Music).
ParticipationLevel (INT): Level of participation (scale of 1-5).
6. Teacher Information Table

Table Name: teacher_information
Description: Contains information about teachers assigned to subjects.
Columns:
TeacherID (INT, Primary Key): Unique identifier for each teacher.
Subject (VARCHAR): Subject taught by the teacher.
YearsOfExperience (INT): Total years of teaching experience.
QualificationLevel (VARCHAR): Highest qualification of the teacher.
*Relationships*
The relationships among the tables are designed to reflect the interconnected nature of the data:

Student Information ↔️ Behavioral and Socioeconomic Factors:
One-to-One relationship: Each student has one set of behavioral and socioeconomic data.
Student Information ↔️ Academic Performance:
One-to-Many relationship: A student can have multiple academic performance records across different subjects and terms.
Student Information ↔️ Performance Prediction Labels:
One-to-One relationship: Each student has one performance prediction label per academic year.
Student Information ↔️ Extracurricular Activities:
One-to-Many relationship: A student can participate in multiple extracurricular activities.
Academic Performance ↔️ Teacher Information (optional):
Many-to-One relationship: Multiple academic performance records can be associated with one teacher.
